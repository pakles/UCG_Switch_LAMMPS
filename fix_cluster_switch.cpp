/* ---------------------------------------------------------------------- */
#include "mpi.h"
#include "modify.h"
#include "memory.h"
#include "update.h"
#include "respa.h"
#include "force.h" //force->numeric(), etc
#include "random_mars.h" //random number generator
#include "fix_cluster_switch.h"
#include "error.h" //error
#include <string.h> //strcmp()

#include "neighbor.h" //neigh->build
#include "neigh_list.h" //class NeighList
#include "neigh_request.h" //neigh->request
#include "atom.h" //per-atom variables
#include "pair.h" //force->pair
#include "bond.h" //force->bond
#include "group.h"//temperature compute
#include "compute.h" //temperature->compute_scalar
#include "comm.h"

#include <fstream> //read input files: rates and mhcorr
#include <string> //read input files: rates and mhcorr
using namespace LAMMPS_NS;
using namespace FixConst; //in fix.h, defines POST_FORCE, etc.

#define MYDEBUG
 #define MAXLINE 1024
//#define MYDEBUG1

/* ---------------------------------------------------------------------- */
FixClusterSwitch::FixClusterSwitch(LAMMPS *lmp, int narg, char **arg) : Fix(lmp,narg,arg)
{
  /*
  USAGE:
     fix  1   all  cluster_switch molID_seed mol_offset  cutoff  15123(seed) rateFreq   1000  rateFile  rates.txt  contactFile  contacts.txt
         [0]  [1]     [2]            [3]       [4]         [5]        [6]       [7]     [8]      [9]        [10]       [11]        [12]
  */
  if (narg < 8) error->all(FLERR,"Illegal cluster_switch command");

  mol_seed = force->inumeric(FLERR,arg[3]);  
  mol_offset = force->inumeric(FLERR,arg[4]); // molIDs that are mol_offset less than molIDs-counted-based-on-rates.txt should be counted toward clustering
  double cutoff = force->numeric(FLERR,arg[5]); //cutoff distance for contact mapping
  
  int seed = force->inumeric(FLERR,arg[6]);

  switchFreq = force->inumeric(FLERR,arg[8]); // how often attempts to switch are made, equivalent to "nevery"
  random = new RanMars(lmp, seed);

  // READ INPUT FILE THAT CONTAINS RATES AND ATOM TYPE INFO
  read_file(arg[10]); // reads the rate file and initializes/populates arrays
  read_contacts(arg[12]); // reads contact map file

  // ERROR CHECKPOINT: make sure pair style has cutoff attribute
  if (force->pair == NULL) error->all(FLERR,"fix cluster_switch requires a pair style");
  if (force->pair->cutsq == NULL) error->all(FLERR,"fix cluster_switch is incompatible with pair style");
  cutsq = cutoff * cutoff; //force->pair->cutsq;
  
  // INITIALIZE FLAGS
  force_reneighbor = 1; //fix requires reneighboring
  next_reneighbor = update->ntimestep + 1; //reneighboring flag, will set later on
  comm_forward = 1; //Flag for forward communication (send local atom data to other procs as ghost particles)
  pack_flag = 0; //This may or may not be used later (to set which information is transmitted to adjacent nodes)
  
  vector_flag = 1;
  size_vector = 7; 
  global_freq = 1;
  extvector = 0;
  time_depend = 1;
  
  // INITIALIZE COUNTERS FOR STATISTICS
  nAttemptsTotal = 0.0;
  nAttemptsON = 0.0;
  nAttemptsOFF = 0.0;
  nSuccessTotal = 0.0;
  nSuccessON = 0.0;
  nSuccessOFF = 0.0;
  nCluster = 0.0;

  // check molecule flag
  if(atom->molecule_flag == 0) error->all(FLERR,"fix cluster_switch requires that atoms have molecule attributes");

  int nmolatoms = 0;
  int maxmol_local = -1;
  int *atypes = atom->type;
  int *molecule = atom->molecule;
  
  for(int i = 0; i < atom->nlocal; i++){
    if(atom->mask[i] & groupbit){
      if(molecule[i] > maxmol_local) maxmol_local = molecule[i];
      for(int j = 0; j < nSwitchTypes; j++){
	if(atypes[i] == atomtypesON[j] || atypes[i] == atomtypesOFF[j]) nmolatoms++;
      }
    }
  }
  MPI_Allreduce(&nmolatoms, &nmol, 1, MPI_INT, MPI_SUM, world);
  MPI_Allreduce(&maxmol_local, &maxmol, 1, MPI_INT, MPI_MAX, world);
  nmol = nmol / nSwitchTypes;
  //printf("Maximum mol number is %d\n",maxmol);

  if(maxmol < 0) error->all(FLERR,"Selected group does not have any mols (fix cluster_switch)");

  allocate_flag = 1;
  allocate();

  //populate mol_restrict list with -1 if restricted and 1 if enabled, then share data across processors
  int *mol_restrict_local = new int[maxmol+1];
  int *mol_state_local = new int[maxmol+1];
  for(int i = 0; i <= maxmol; i++){
    mol_restrict_local[i] = -1;
    mol_state_local[i] = -1;
  }
  
  for(int i = 0; i < atom->nlocal; i++){
    if(atom->mask[i] & groupbit){
      int molID = molecule[i];

      for(int j = 0; j < nSwitchTypes; j++){
	if(atypes[i] == atomtypesON[j] && mol_state_local[molID] == -1) {
	  mol_state_local[molID] = 1;
	  if(molID != mol_seed && molID != (mol_seed - mol_offset)){
	    mol_restrict_local[molID] = 1;
	  }
	}
	else if(atypes[i] == atomtypesOFF[j] && mol_state_local[molID] == -1) {
	  mol_state_local[molID] = 0;
	  if(molID != mol_seed && molID != (mol_seed - mol_offset)){
	    mol_restrict_local[molID] = 1;
	  }
	}
      }
    }
  }

  MPI_Allreduce(mol_restrict_local, mol_restrict, maxmol+1, MPI_INT, MPI_MAX, world);
  MPI_Allreduce(mol_state_local, mol_state, maxmol+1, MPI_INT, MPI_MAX, world); 

  delete [] mol_restrict_local;
  delete [] mol_state_local;
  
  check_arrays();
  
}

/* ---------------------------------------------------------------------- */
FixClusterSwitch::~FixClusterSwitch()
{
  delete random;

  memory->destroy(atomtypesON);
  memory->destroy(atomtypesOFF);
  memory->destroy(mol_restrict);
  memory->destroy(mol_atoms);
  memory->destroy(mol_state);
  memory->destroy(mol_cluster);
  memory->destroy(mol_accept);
  memory->destroy(contactMap);
}


/* ---------------------------------------------------------------------- */
void FixClusterSwitch::read_file(char *file)
{
  // open file on proc 0

  FILE *fp;
  if (comm->me == 0) {
    fp = force->open_potential(file);
    if (fp == NULL) {
      char str[128];
      sprintf(str,"Cannot open rates file (fix cluster_switch) called %s",file);
      error->one(FLERR,str);
    }
  }

  // read each line out of file, skipping blank lines or leading '#'
  // store line of params if all 3 element tags are in element list

  int n,nwords,lineNum,eof;
  char line[MAXLINE],*ptr;
  char **words = new char*[100];

  eof = lineNum = 0;

  while (1) {
    if (comm->me == 0) {
      ptr = fgets(line,MAXLINE,fp);
      if (ptr == NULL) {
        eof = 1;
        fclose(fp);
      } else n = strlen(line) + 1;
    }
    MPI_Bcast(&eof,1,MPI_INT,0,world);
    if (eof) break;
    MPI_Bcast(&n,1,MPI_INT,0,world);
    MPI_Bcast(line,n,MPI_CHAR,0,world);

    lineNum++;

    // strip comment, skip line if blank

    if ((ptr = strchr(line,'#'))) *ptr = '\0';
    nwords = atom->count_words(line);
    if (nwords == 0) continue;

    // words = ptrs to all words in line

    nwords = 0;
    words[nwords++] = strtok(line," \t\n\r\f");
    while ((words[nwords++] = strtok(NULL," \t\n\r\f"))) continue;

    if(lineNum == 1) {
      probON  = atof(words[0]);
      if(probON > 1.0) {
	char str[128];
	sprintf(str,"Incorrect probability in rates.txt files (fix cluster_switch)");
	error->one(FLERR,str);
      }
      probOFF = 1.0 - probON;
    }
    else if (lineNum == 2) {
      nSwitchTypes = atoi(words[0]);
      if(nSwitchTypes > atom->ntypes) {
	char str[128];
	sprintf(str,"Incorrect number of atom switching types (fix cluster_switch)");
	error->one(FLERR,str);
      }
      allocate_flag = 2;
      allocate();
    }
    else if (lineNum == 3) {
      for(int i = 0; i < nSwitchTypes; i++) atomtypesON[i] = atoi(words[i]);
    }
    else if (lineNum == 4) {
      for(int i = 0; i < nSwitchTypes; i++) atomtypesOFF[i] = atoi(words[i]);
    }
  }
  delete [] words;


}


/* ---------------------------------------------------------------------- */
void FixClusterSwitch::read_contacts(char *file)
{
  // open file on proc 0

  FILE *fp;
  if (comm->me == 0) {
    fp = force->open_potential(file);
    if (fp == NULL) {
      char str[128];
      sprintf(str,"Cannot open contact file (fix cluster_switch) called %s",file);
      error->one(FLERR,str);
    }
  }

  // read each line out of file, skipping blank lines or leading '#'
  // store line of params if all 3 element tags are in element list

  int n,nwords,lineNum,eof;
  char line[MAXLINE],*ptr;
  char **words = new char*[100];

  eof = lineNum = 0;

  while (1) {
    if (comm->me == 0) {
      ptr = fgets(line,MAXLINE,fp);
      if (ptr == NULL) {
        eof = 1;
        fclose(fp);
      } else n = strlen(line) + 1;
    }
    MPI_Bcast(&eof,1,MPI_INT,0,world);
    if (eof) break;
    MPI_Bcast(&n,1,MPI_INT,0,world);
    MPI_Bcast(line,n,MPI_CHAR,0,world);

    lineNum++;

    // strip comment, skip line if blank

    if ((ptr = strchr(line,'#'))) *ptr = '\0';
    nwords = atom->count_words(line);
    if (nwords == 0) continue;

    // words = ptrs to all words in line

    nwords = 0;
    words[nwords++] = strtok(line," \t\n\r\f");
    while ((words[nwords++] = strtok(NULL," \t\n\r\f"))) continue;

    if(lineNum == 1) {
      nContactTypes  = atoi(words[1]);
      //      printf("Num Contacts %d\n",nContactTypes);
    }
    else if (lineNum == 2) {
      nAtomsPerContact = atoi(words[1]);
      //      printf("Atoms per Contact %d\n",nAtomsPerContact);
      allocate_flag = 3;
      allocate();
    }
    else {
      int lineOffset = lineNum - 3;
      int i = lineOffset / nAtomsPerContact; // take advantage of integer division
      int j = lineOffset - i*nAtomsPerContact;
      contactMap[i][j][0] = atoi(words[0]);
      contactMap[i][j][1] = atoi(words[1]);
      //      printf("Contacts [%d][%d] are %d and %d\n",i,j,contactMap[i][j][0],contactMap[i][j][1]);
    }
  }
  delete [] words;


}


/* ---------------------------------------------------------------------- */
void FixClusterSwitch::init()
{
  ngroup = group->count(igroup);
  //groupbit = group->bitmask[igroup];
    
  if(strstr(update->integrate_style,"respa")) nlevels_respa = ((Respa *) update->integrate)->nlevels;
  
  //need a full neighbor list, built whenever re-neighboring occurs//
  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->fix  = 1;
  neighbor->requests[irequest]->half = 0; //default is half list
  neighbor->requests[irequest]->full = 1;
  neighbor->requests[irequest]->occasional = 1;

}

void FixClusterSwitch::init_list(int id, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */
void FixClusterSwitch::allocate()
{
  if (allocate_flag == 1)
    {
      memory->create(mol_restrict,maxmol+1,"fix:mol_restrict");
      memory->create(mol_state,maxmol+1,"fix:mol_state");
      memory->create(mol_accept,maxmol+1,"fix:mol_accept");
      memory->create(mol_cluster,maxmol+1,"fix:mol_cluster");
      memory->create(mol_atoms,maxmol+1,nSwitchTypes,"fix:mol_atoms");

      //no memory; initialize array such that every state is -1 to start
      for(int i = 0; i <= maxmol; i++){
	mol_restrict[i] = -1;
	mol_state[i] = -1;
	mol_accept[i] = -1;
	for(int j = 0; j < nSwitchTypes; j++){
	  mol_atoms[i][j] = -1;
	}
      }
    }

  else if(allocate_flag == 2)
    {
      memory->create(atomtypesON,nSwitchTypes,"fix:atomtypesON");
      memory->create(atomtypesOFF,nSwitchTypes,"fix:atomtypesOFF");
    }

  else if(allocate_flag == 3)
    {
      memory->create(contactMap,nContactTypes,nAtomsPerContact,2,"fix:contactMap");
    }

}

/* ---------------------------------------------------------------------- */
int FixClusterSwitch::setmask()
{
  int mask=0;
  mask |= PRE_FORCE;
  mask |= PRE_FORCE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */
void FixClusterSwitch::setup_pre_force(int vflag)
{
  check_cluster();
  attempt_switch();
}

/* ---------------------------------------------------------------------- */
void FixClusterSwitch::setup_pre_force_respa(int vflag, int ilevel)
{
  if(ilevel < nlevels_respa-1) return;
  setup_pre_force(vflag);
}

void FixClusterSwitch::pre_force_respa(int vflag, int ilevel, int)
{
  if(ilevel < nlevels_respa-1) return;
  pre_force(vflag);
}

void FixClusterSwitch::pre_force(int vflag)
{
  if (switchFreq == 0) return;
  if (update->ntimestep % switchFreq) return;
  check_cluster();
  attempt_switch();
}

void FixClusterSwitch::normalize(double *a)
{
  double mag=0.0;
  for(int k=0;k<3;k++) mag += a[k]*a[k];
  if(mag<=0) error->all(FLERR,"Normalization error in cluster_switch");
  mag=sqrt(mag);
  for(int k=0;k<3;k++) a[k] /= mag;
}

void FixClusterSwitch::computecross(double *a, double *b1, double *b2)
{
  a[0] = b1[1]*b2[2] - b1[2]*b2[1];
  a[1] = b1[2]*b2[0] - b1[0]*b2[2];
  a[2] = b1[0]*b2[1] - b1[1]*b2[0];
}
 

void FixClusterSwitch::check_arrays()
{
  for(int i = 0; i <= maxmol; i++){
    //printf("For mol %d, restrict flag is %d and state flag is %d on proc %d\n", i, mol_restrict[i], mol_state[i], comm->me);
    if(mol_restrict[i] == 1 && !(mol_state[i] == 1 || mol_state[i] == 0)) error->all(FLERR,"Communication of mol_state inconsistent: fix cluster_switch");
  }
}

/* ---------------------------------------------------------------------- */
// Sends local atom data toward neighboring procs (into ghost atom data)
// Reverse sends ghost data to local data (e.g., forces)
int FixClusterSwitch::pack_forward_comm(int n, int *list, double *buf, int pbc_flag, int *pbc)
{
  int m;

  //  if(pack_flag == 1)
  //    for(m = 0; m < n; m++) buf[m] = mol_restrict[list[m]];
  //  if(pack_flag == 2)
  //    for(m = 0; m < n; m++) buf[m] = mol_state[list[m]];
  //  if(pack_flag == 3)
  //    for(m = 0; m < n; m++) buf[m] = mol_accept[list[m]];
  //  if(pack_flag == 4)
  for(m = 0; m < n; m++) buf[m] = atom->type[list[m]];

  return m;
}

/* ---------------------------------------------------------------------- */
void FixClusterSwitch::unpack_forward_comm(int n, int first, double *buf)
{
  int i, m;

  //  if(pack_flag == 1)
  //    for(m = 0, i = first; m < n; m++, i++) mol_restrict[i] = buf[m];
  //  if(pack_flag == 2)
  //    for(m = 0, i = first; m < n; m++, i++) mol_state[i] = buf[m];
  //  if(pack_flag == 1)
  //    for(m = 0, i = first; m < n; m++, i++) mol_accept[i] = buf[m];
  //  if(pack_flag == 2)
  for(m = 0, i = first; m < n; m++, i++) atom->type[i] = buf[m];
}

/* ---------------------------------------------------------------------- */
double FixClusterSwitch::memory_usage()
{
  return 0;
}


void FixClusterSwitch::check_cluster()
{
  int *mask = atom->mask;
  tagint *molecule = atom->molecule;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;

  // invoke neighbor list (full)
  neighbor->build_one(list);
  int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

  int *mol_cluster_local = new int[maxmol+1];
  for(int i = 0; i <= maxmol; i++){
    mol_cluster_local[i] = -1;
    mol_cluster[i] = -1;

  }

  //preset mol_seed and mol_offset
  mol_cluster_local[mol_seed] = mol_cluster[mol_seed] = mol_seed;
  mol_cluster_local[mol_seed-mol_offset] = mol_cluster[mol_seed-mol_offset] = mol_seed;

  for(int ii = 0; ii < inum; ii++) {
    int i = ilist[ii];
    if(mask[i] & groupbit) {
      int molID = molecule[i];
      mol_cluster_local[molID] = molID;
    }
  }

  //make sure mol_offset criteria is also accounted for
  for(int ii = 0; ii < inum; ii++) {
    int i = ilist[ii];
    if(mask[i] & groupbit) {
      int molID = molecule[i];
      if(mol_state[molID] == 0 || mol_state[molID] == 1){
	mol_cluster_local[molID-mol_offset] = molID;
      }
    }
  }


  MPI_Allreduce(mol_cluster_local,mol_cluster,maxmol+1,MPI_INT,MPI_MAX,world);

  // loop until no more changes to mol_cluster (local copy on mol_cluster_local) on any proc
  double **x = atom->x;

  int change,done,anychange;
  double xtmp, ytmp, ztmp, delx, dely, delz, rsq;
  int *jlist, jnum;

  while (1) {

    //update local copy
    for(int i = 0; i <= maxmol; i++) mol_cluster_local[i] = mol_cluster[i];

    change = 0;
    while (1) {
      done = 1;
      for(int ii = 0; ii < inum; ii++){
	int i = ilist[ii];
	int i_molID = molecule[i];
	int itype = type[i];
	if(!(mask[i] & groupbit)) continue;

	xtmp = x[i][0];
	ytmp = x[i][1];
	ztmp = x[i][2];
	jlist = firstneigh[i];
	jnum = numneigh[i];

	for(int jj = 0; jj < jnum; jj++){
	  int j = jlist[jj];
	  int j_molID = molecule[j];
	  int jtype = type[j];
	  j &= NEIGHMASK;
	  if(!(mask[j] & groupbit)) continue;
	  if(mol_cluster_local[i_molID] == mol_cluster_local[j_molID]) continue;

	  // first check to see if valid contact types
	  //	  printf("Checking types %d and %d now\n",itype,jtype);

	  int contactFlag = -1;
	  for(int m = 0; m < nContactTypes; m++){
	    if(contactFlag > -1) break;
	    for(int n = 0; n < nAtomsPerContact; n++){
	      if(contactFlag > -1) break;
	      if(contactMap[m][n][0] == itype && contactMap[m][n][1] == jtype) contactFlag = 1;
	    }
	  }

	  if(contactFlag == 1) {
	    delx = xtmp - x[j][0];
	    dely = ytmp - x[j][1];
	    delz = ztmp - x[j][2];
	    rsq = delx*delx + dely*dely + delz*delz;

	    //then check if distance criteria is valid and update mols and offset mols
	    //next version needs to incorporate explicit contact maps
	    if(rsq < cutsq) {
	      int newID_v1 = MIN(mol_cluster_local[i_molID],mol_cluster_local[j_molID]);
	      int newID_v2 = newID_v1;
	      int newID_v3 = newID_v2;

	      //if mol is a switchable mol, check -mol_offset; else, check mol_offset
	      if(mol_state[i_molID] == 0 || mol_state[i_molID] == 1) newID_v2 = MIN(mol_cluster_local[i_molID-mol_offset],newID_v1);
	      else newID_v2 = MIN(mol_cluster_local[i_molID+mol_offset],newID_v1);

	      if(mol_state[j_molID] == 0 || mol_state[j_molID] == 1) newID_v3 = MIN(mol_cluster_local[j_molID-mol_offset],newID_v2);
	      else newID_v3 = MIN(mol_cluster_local[j_molID+mol_offset],newID_v2);
	    
	      //now update cluster ID of i, j
	      mol_cluster_local[i_molID] = mol_cluster_local[j_molID] = newID_v3;

	      //and update cluster ID of offset mols to be consistent
	      if(mol_state[i_molID] == 0 || mol_state[i_molID] == 1) mol_cluster_local[i_molID-mol_offset] = newID_v3;
	      else mol_cluster_local[i_molID+mol_offset] = newID_v3;

	      if(mol_state[j_molID] == 0 || mol_state[j_molID] == 1) mol_cluster_local[j_molID-mol_offset] = newID_v3;
	      else mol_cluster_local[j_molID+mol_offset] = newID_v3;

	      done = 0;
	    }
	  }
	}
      }
      if (!done) change = 1;
      if (done) break;
    }

    MPI_Allreduce(&change,&anychange,1,MPI_INT,MPI_MAX,world);
    MPI_Allreduce(mol_cluster_local,mol_cluster,maxmol+1,MPI_INT,MPI_MIN,world); // send min cluster ID to global array
    if (!anychange) break;
  }

  // now switch mol_restrict flags based on cluster ID of mol_seed (mol_cluster should be copied beforehand)
  int clusterID = mol_cluster[mol_seed];
  nCluster = 0.0;
  for(int i = 0; i <= maxmol; i++){
    if(mol_cluster[i] != -1){
      // if this is a switchable mol, mol_restrict should be updated
      if(mol_state[i] == 0 || mol_state[i] == 1) {
	if(mol_cluster[i] == clusterID) {
	  mol_restrict[i] = -1;
	}
	else mol_restrict[i] = 1;
      }
      if(mol_cluster[i] == clusterID) nCluster += 1.0;
    }
  }

  delete [] mol_cluster_local;

}

void FixClusterSwitch::attempt_switch()
{
  int *mask = atom->mask;
  tagint *molecule = atom->molecule; // molecule[i] = mol IDs for atom tag i
  int nlocal = atom->nlocal;

  //first gather unique molIDs on this processor
  hash = new std::map<tagint,int>();

  for(int i = 0; i < nlocal; i++){
    if(mask[i] & groupbit){
      if(hash->find(molecule[i]) == hash->end()) {
	(*hash)[molecule[i]] = 1;
	//printf("molecule[i] for i = %d prints out molID %d\n",i,molecule[i]);
      }
    }
  }
  int nmol_local = hash->size();
  //printf("Total number of mols on proc %d is %d\n",comm->me,nmol_local);

  //no memory; initialize array such that every state is -1 to start
  for(int i = 0; i < nmol; i++){
    for(int j = 0; j < nSwitchTypes; j++){
      mol_atoms[i][j] = -1;
    }
  }

  //local copy of mol_accept (this should be zero'd at every attempt
  int *mol_accept_local = new int[maxmol+1];
  for(int i = 0; i <= maxmol; i++){
    mol_accept_local[i] = -1;
    mol_accept[i] = -1;
    for(int j = 0; j < nSwitchTypes; j++) mol_atoms[i][j] = -1;
  }

  std::map<tagint,int>::iterator pos;
  for (pos = hash->begin(); pos != hash->end(); ++pos){
    tagint mID = pos->first;
    int confirmflag;
    //    printf("molID is %d on proc %d\n",mID,comm->me);
    //    mol_restrict[n] = mID;
    //    if(mol_restrict[mID] != 1) error->all(FLERR,"Current molID in fix cluster_switch is not valid!");
    if(mol_restrict[mID] == 1) confirmflag = confirm_molecule(mID); //checks if this proc should be decision-maker
    else confirmflag = 0;

    if(mol_accept_local[mID] == -1 && confirmflag != 0)  mol_accept_local[mID] = switch_flag(mID);
  }

  // communicate accept flags across processors...
  MPI_Allreduce(mol_accept_local, mol_accept, maxmol+1, MPI_INT, MPI_MAX, world);
  gather_statistics(); //keep track of MC statistics
  check_arrays();

  //now perform switchings on each proc
  int *atype = atom->type;
  for(int i = 0; i <= maxmol; i++){
    //if acceptance flag is turned on
    if(mol_accept[i] == 1){
      for(int j = 0; j < nSwitchTypes; j++){
	int tagi = mol_atoms[i][j];
	//if originally in OFF state
	if(mol_state[i] == 0 && tagi > -1){
	  atype[tagi] = atomtypesON[j];
	}
	//if originally in ON state
	else if(mol_state[i] == 1 && tagi > -1){
	  atype[tagi] = atomtypesOFF[j];
	}
      }
      //update mol_state
      if(mol_state[i] == 0) mol_state[i] = 1;
      else if(mol_state[i] == 1) mol_state[i] = 0;
    }
  } 

  //communicate changed types
  //  pack_flag = 4;
  comm->forward_comm_fix(this);


  delete [] mol_accept_local;
  delete hash;

}

int FixClusterSwitch::confirm_molecule( tagint molID )
{
  tagint *molecule = atom->molecule;
  int *atype = atom->type;
  double sumState = 0.0;
  double decisionBuffer = nSwitchTypes/2.0 - 1.0 + 0.01; // just to ensure that the proc with the majority of the switching types makes the switching decision
  for(int i = 0; i < atom->nlocal; i++){
    //    printf("Checking molecule[i] with tagid %d against molID %d on proc %d\n",molecule[i],molID,comm->me);
    if(molecule[i] == molID){
      int itype = atype[i];
      //      printf("Found a matching molecule! Now checking against nSwitchType %d using current itype %d on proc %n\n",nSwitchTypes,itype,comm->me);
      for(int j = 0; j < nSwitchTypes; j++){
	if(itype == atomtypesON[j]){
	  mol_atoms[molID][j] = i;
	  sumState += 1.0;
	}
	else if(itype == atomtypesOFF[j]){
	  mol_atoms[molID][j] = i;
	  sumState -= 1.0;
	}
      }
    }

  }
  //  printf("Current sumState is %d with molID %d and n %d\n", sumState, molID, n);
  if(sumState < (decisionBuffer * -1)) return -1;
  else if(sumState > decisionBuffer) return 1;
  else return 0;
}


int FixClusterSwitch::switch_flag( int molID )
{
  int state = mol_state[molID];
  double checkProb;

  // if current state is OFF (0), then use probability to turn ON
  if(state == 0) {
    checkProb = probON;
  }
  else {
    checkProb = probOFF;
  }
  
  double rand = random->uniform();
  if(rand < checkProb){
    return 1;
  }
  else
    return 0;
}

double FixClusterSwitch::compute_vector(int n)
{
  if (n == 0) return nAttemptsTotal;
  if (n == 1) return nSuccessTotal;
  if (n == 2) return nAttemptsON;
  if (n == 3) return nAttemptsOFF;
  if (n == 4) return nSuccessON;
  if (n == 5) return nSuccessOFF;
  if (n == 6) return nCluster;
  return 0.0;
}

void FixClusterSwitch::gather_statistics()
{
  //gather these stats before mol_state is updated
  double dt_AttemptsTotal = 0.0, dt_AttemptsON = 0.0, dt_AttemptsOFF = 0.0;
  double dt_SuccessTotal = 0.0, dt_SuccessON = 0.0, dt_SuccessOFF = 0.0;
  for(int i = 0; i <= maxmol; i++) {
    if(mol_restrict[i] == 1){
      dt_AttemptsTotal += 1.0;
      if(mol_state[i] == 0) {
	dt_AttemptsON += 1.0;
	if(mol_accept[i] == 1){
	  dt_SuccessTotal += 1.0;
	  dt_SuccessON += 1.0;
	}
      }
      else if(mol_state[i] == 1){
	dt_AttemptsOFF += 1.0;
	if(mol_accept[i] == 1){
	  dt_SuccessTotal += 1.0;
	  dt_SuccessOFF += 1.0;
	}
      }
    }
  }

  //now update
  nAttemptsTotal += dt_AttemptsTotal;
  nAttemptsON += dt_AttemptsON;
  nAttemptsOFF += dt_AttemptsOFF;
  nSuccessTotal += dt_SuccessTotal;
  nSuccessON += dt_SuccessON;
  nSuccessOFF += dt_SuccessOFF;
   
}
