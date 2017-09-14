/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Alex Pak (University of Chicago)
------------------------------------------------------------------------- */

#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "fix_reservoir_bias.h"
#include "math_const.h"
#include "atom.h"
#include "input.h"
#include "domain.h"
#include "lattice.h"
#include "update.h"
#include "modify.h"
#include "output.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "comm.h"
#include "group.h"

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;


/* ---------------------------------------------------------------------- */

FixReservoirBias::FixReservoirBias(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg)
{
  if (narg < 10) error->all(FLERR,"Illegal fix indent command");

  /*
    USAGE: fix 1 GAG reservoir/bias 1.0 0.5 2.0 400.0 1 7.5 40.0
    arg[3] = force constant in kcal/mol/atom/A^2
    arg[4] = target density in atom/A^2
    arg[5] = threshold density in atom/A^2
    arg[6] = control region thickness in A
    arg[7] = frequency of applied bias in timesteps
    arg[8] = sigma in Gaussian broadening function in A
    arg[9] = bin size of density profile in A
   */

  MPI_Comm_rank(world,&me);
  // printf("This rank of this proc is %d\n", me);

  scalar_flag = 1;
  vector_flag = 1;
  size_vector = 5; //force x. y. z. rSL, nCR
  global_freq = 1;
  extscalar = 1;
  extvector = 1;
  rigid_flag = 1; //integrates rigid bodies

  comm_forward = 1; // need to transmit type data across procs (when switching atom types)

  // check molecule flag
  if(atom->molecule_flag == 0) error->all(FLERR,"Fix reservoir/bias requires atoms with the molecule attribute");
  int nmolatoms = 0;
  int maxmol_local = -1;
  int *atypes = atom->type;
  int *molecule = atom->molecule;

  // preset atom types for switching
  nSwitchTypes = 4;
  atomtypesON[0] = 158;
  atomtypesON[1] = 159;
  atomtypesON[2] = 160;
  atomtypesON[3] = 161;
  atomtypesOFF[0] = 168;
  atomtypesOFF[1] = 169;
  atomtypesOFF[2] = 170;
  atomtypesOFF[3] = 171;
  
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
  nmol /= nSwitchTypes;


  // initialize values that are user defined
  k = force->numeric(FLERR,arg[3]);
  n0 = force->numeric(FLERR,arg[4]);
  nSL = force->numeric(FLERR,arg[5]);
  rCR = force->numeric(FLERR,arg[6]);
  nevery = force->inumeric(FLERR,arg[7]);
  w = force->numeric(FLERR,arg[8]);
  binwidth = force->numeric(FLERR,arg[9]);
  
  w6 = 6*w;
  w6sq = w6*w6;
  invbinwidth = 1.0/binwidth;

  // initialize values that need to be calculated on the fly
  rSL = 0.0;
  nCR = 0.0;
  nummaxbin = 100; //guess a size
  bincoord = NULL;
  numarea = NULL;
  numarea_all = NULL;
  mol_preferred_state = NULL;
  // fp = NULL;

  // read options from end of input line
  // options(narg-4,&arg[4]);

  indenter_flag = 0;
  align_flag = 0;
  align2_flag = 0;
  indenter[0] = indenter[1] = indenter[2] = indenter[3] = indenter[4] = indenter[5] = 0.0;
  center[0] = center[1] = center[2] = 0.0;


  //  if(me == 0) {
  //    fp = fopen("print_density.dat","w+");
  //    if(fp == NULL) {
  //      error->one(FLERR,"File print_density.dat in reservoir/bias not open!\n");
  //    }
  //  }

}

/* ---------------------------------------------------------------------- */

FixReservoirBias::~FixReservoirBias()
{
  memory->destroy(bincoord);
  memory->destroy(numarea);
  memory->destroy(numarea_all);

  memory->destroy(mol_preferred_state);
  //delete [] center;
  //delete [] indenter;
  //delete [] indenter_all;
  //  if(fp && me == 0) fclose(fp);
}

/* ---------------------------------------------------------------------- */

int FixReservoirBias::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixReservoirBias::init()
{
    memory->create(bincoord,nummaxbin,"reservoir/bias:bincoord");
    memory->create(numarea,nummaxbin,"reservoir/bias:numarea");
    memory->create(numarea_all,nummaxbin,"reservoir/bias:numarea_all");

    for(int i = 0; i < nummaxbin; i++){
      bincoord[i] = i*binwidth + 0.5*binwidth;
      //      printf("Bincoord[i] is %f\n",bincoord[i]);
    }

    //initialize values to zero
    for(int i = 0; i < nummaxbin; i++){
      numarea[i] = 0.0;
      numarea_all[i] = 0.0;
    }

    memory->create(mol_preferred_state,maxmol+1,"reservoir/bias:mol_state");

    for(int i = 0; i < maxmol+1; i++){
      mol_preferred_state[i] = 0;
    }
    

}

/* ---------------------------------------------------------------------- */

void FixReservoirBias::setup(int vflag)
{
  //if (strstr(update->integrate_style,"verlet")) post_force(vflag);
}


/* ---------------------------------------------------------------------- */

void FixReservoirBias::post_force(int vflag)
{
  // indenter values, 0 = energy, 1-3 = force components
  // wrap variable evaluations with clear/add

  if(update->ntimestep % nevery) return;

  indenter_flag = 0;
  align_flag = 0;
  align2_flag = 0;
  indenter[0] = indenter[1] = indenter[2] = indenter[3] = indenter[4] = indenter[5] = 0.0;

  // center of domain
  center[0] = domain->boxlo[0] + domain->prd_half[0];
  center[1] = domain->boxlo[1] + domain->prd_half[1];
  center[2] = domain->boxlo[2] + domain->prd_half[2];

  nCR = 0.0;
  rSL = 0.0;

  //printf("The value of center0 is %f\n", center[0]);
  //printf("The value of center1 is %f\n", center[1]);
  //printf("The value of boxlo: is %f and %f\n", domain->boxlo[0], domain->boxlo[1]);
  //printf("The value of prdhalf is %f and %f\n", domain->prd_half[0], domain->prd_half[1]);

  //setup all the density bins
  double refR;
  int maxBinTemp;

  refR = MIN(domain->prd_half[0], domain->prd_half[1]);
  //  printf("The value of refR is %f\n", refR);

  maxBinTemp = static_cast<int>(refR * invbinwidth);

  //  printf("The value of maxBinTemp is %d\n", maxBinTemp);
  //  printf("The value of nummaxbin is %d\n", nummaxbin);
  
  if(maxBinTemp > nummaxbin){
    //memory->destroy(bincoord);
    //memory->destroy(numarea);
    //memory->destroy(numarea_all);

    nummaxbin = maxBinTemp + 10;

    memory->grow(bincoord,nummaxbin,"reservoir/bias:bincoord");
    memory->grow(numarea,nummaxbin,"reservoir/bias:numarea");
    memory->grow(numarea_all,nummaxbin,"reservoir/bias:numarea_all");

    for(int i = 0; i < nummaxbin; i++){
      bincoord[i] = i*binwidth + 0.5*binwidth;
      //      printf("Bincoord[i] is %f\n",bincoord[i]);
    }
  }

  //initialize values to zero
  for(int i = 0; i < nummaxbin; i++){
    numarea[i] = 0.0;
    numarea_all[i] = 0.0;
  }

  //  printf("The value of maxBinTemp is %d\n", maxBinTemp);
  //  printf("The value of nummaxbin is %d\n", nummaxbin);

  calc_radial_density();

  print_density();

  apply_restraint();

  activate_assembly_status();
  
}


/* ---------------------------------------------------------------------- 
   After density bins are setup, calculate the number density
   on each local proc and then sum them up using Allgather.
   Finally, determine the radius of the S-L interface
 ---------------------------------------------------------------------- */
void FixReservoirBias::calc_radial_density()
{


  //assumes cylindrical for now
  double **x = atom->x; //atom positions on this proc
  double **f = atom->f; //atom forces
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  int rbin;
  double delx, dely, delz, r, rin, rout, dA; //inner, outer radius and area of slice
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      delx = x[i][0] - center[0];
      dely = x[i][1] - center[1];
      delz = 0.0;
      domain->minimum_image(delx,dely,delz);
      r = sqrt(delx*delx + dely*dely + delz*delz);

      rbin = static_cast<int> (r * invbinwidth);
      rbin = MAX(rbin,0);
      rbin = MIN(rbin,nummaxbin-1);
      
      rin = rbin * binwidth;
      rout = (rbin+1) * binwidth;
      dA = MY_PI * (rout*rout - rin*rin); 
      
      numarea[rbin] += 1.0/dA;
      
    }
  }
  
  //  for(int i = 0; i < nummaxbin; i++){
  //    printf("numarea[i] is %f\n",numarea[i]);
  //  }

  // now sum all numarea arrays and store in numarea_all array
  // only do this once
  if(align_flag == 0){
    MPI_Allreduce(numarea, numarea_all, nummaxbin, MPI_DOUBLE, MPI_SUM, world);
    align_flag = 1;
  }
  //  for(int i = 0; i < nummaxbin; i++){
  //    printf("numarea_all[i] is %f\n",numarea_all[i]);
  //  }
  
  //now set rSL based on nSL and calculate nCR
  int flagSumDens = 0;
  double nSampled = 0.0;
  for(int i = 0; i < nummaxbin; i++){

    if(flagSumDens == 1 && bincoord[i] < (rSL + rCR)){
      nSampled += 1.0;
      nCR += numarea_all[i];
    }

    if(numarea_all[i] < nSL && flagSumDens == 0){
      rSL = bincoord[i];
      flagSumDens = 1;
    }
  }

  nCR /= nSampled;
  //printf("Calculated rSL as %f and nCR as %f\n", rSL, nCR);
}

/* ---------------------------------------------------------------------- 
   After density is calculated and parameters of restraining bias are
   found, this method will apply the cylindrically symmetric bias to
   all atoms in the group
 ---------------------------------------------------------------------- */
void FixReservoirBias::apply_restraint()
{

  //modify->clearstep_compute();

  //assumes cylindrical for now
  double **x = atom->x; //atom positions on this proc
  double **f = atom->f; //atom forces
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int *molecule = atom->molecule;
  int *mol_preferred_state_local = new int[maxmol+1];

  double delx,dely,delz,r,dr,dn,G,fmag,fx,fy,fz,dE;

  
  for(int i = 0; i < maxmol+1; i++){
    mol_preferred_state_local[i] = 0;
  }

  // calculate forces and determine if this mol should be ON or OFF
  for (int i = 0; i < nlocal; i++) {
    if (mask[i] & groupbit) {
      delx = x[i][0] - center[0];
      dely = x[i][1] - center[1];
      delz = 0;
      domain->minimum_image(delx,dely,delz);
      r = sqrt(delx*delx + dely*dely + delz*delz);
      dr = r - (rCR + rSL);
      dn = nCR - n0;
      G = exp(-dr*dr/(2*w*w));
      fmag = k*dn*G;

      if(dr*dr <= w6sq) continue; 

      dE = k * dn * (MY_PIS/MY_SQRT2) * w * erfc( -dr / (MY_SQRT2*w) );

      fx = delx*fmag/r;
      fy = dely*fmag/r;
      fz = delz*fmag/r;
      f[i][0] += fx;
      f[i][1] += fy;
      f[i][2] += fz;
      indenter[0] -= dE; 
      indenter[1] -= fx;
      indenter[2] -= fy;
      indenter[3] -= fz;

      // determine if this molecule should be ON or OFF based on majority within rSL+rCR
      int molID = molecule[i];
      if(dr < 0.0) mol_preferred_state_local[molID]++;
      else mol_preferred_state_local[molID]--;
    }
  }
  indenter[4] = rSL;

  //  printf("The value of rSL and indenter[4] are %f and %f\n",rSL,indenter[4]);
  indenter[5] = nCR;

  // now communicate mol preferred state across procs
  if(align2_flag == 0){
    MPI_Allreduce(mol_preferred_state_local,mol_preferred_state,maxmol+1,MPI_INT,MPI_SUM,world);
    align2_flag = 1;
  }

  delete [] mol_preferred_state_local;

  //modify->addstep_compute(update->ntimestep + nevery);
}

/* ---------------------------------------------------------------------- 
   After restraints are applied, checks location of each molecule
   and determines if within rSL + rCR. If so, assembly status is
   turned on and appropriate atom types are flipped. If not,
   assembly status is turned off and appropriate atom types are flipped.
 ---------------------------------------------------------------------- */
void FixReservoirBias::activate_assembly_status()
{
  // first, check mol_state and determine if switching needs to occur
  int *molecule = atom->molecule;
  int *atypes = atom->type;
  int change_flag = 0;

  for(int i = 0; i < atom->nlocal; i++){
    if(atom->mask[i] & groupbit){

      int molID = molecule[i];

      // if this Gag should be activated
      if(mol_preferred_state[molID] > 0){
	for(int j = 0; j < nSwitchTypes; j++){
	  if(atypes[i] == atomtypesOFF[j]) {
	    atypes[i] = atomtypesON[j];
	    change_flag = 1;
	  }
	}
      }

      // otherwise, make sure this Gag is turned off
      else{
	for(int j = 0; j < nSwitchTypes; j++){
	  if(atypes[i] == atomtypesON[j]) {
	    atypes[i] = atomtypesOFF[j];
	    change_flag = 1;
	  }
	}
      }
    }
  }

  // now communicate new atom types across procs
  if(change_flag == 1){
    if(me == 0)     printf("Atom switch just happened!\n");
    comm->forward_comm_fix(this);
  }

}

/* ----------------------------------------------------------------------
   energy of indenter interaction
------------------------------------------------------------------------- */

double FixReservoirBias::compute_scalar()
{
  // only sum across procs one time
  if(indenter_flag == 0) {
    MPI_Allreduce(indenter,indenter_all,6,MPI_DOUBLE,MPI_SUM,world);
    indenter_flag = 1;
  }
  indenter_all[4] = rSL;
  indenter_all[5] = nCR;

  return indenter_all[0];
}

/* ----------------------------------------------------------------------
   components of force on indenter
------------------------------------------------------------------------- */

double FixReservoirBias::compute_vector(int n)
{
  // only sum across procs one time

  if (indenter_flag == 0) {
    MPI_Allreduce(indenter,indenter_all,6,MPI_DOUBLE,MPI_SUM,world);
    indenter_flag = 1;
  }
  indenter_all[4] = rSL;
  indenter_all[5] = nCR;

  return indenter_all[n+1];
}


/* ---------------------------------------------------------------------- 
   Prints density to file for debugging and checking purposes
 ---------------------------------------------------------------------- */
void FixReservoirBias::print_density()
{
  //  if(fp && me == 0){
  if(me == 0) {
    FILE *fp;
    fp = fopen("print_density.dat","w+");
    if(fp == NULL) {
      error->one(FLERR,"File print_density.dat in reservoir/bias not open!\n");
    }
    //clearerr(fp);
    fprintf(fp,"# Radius(A) NumDens(#/A^2)\n");
    for(int i = 0; i < nummaxbin; i++){
      fprintf(fp,"%f %f\n",bincoord[i],numarea_all[i]);
    }
    if(ferror(fp))
      error->one(FLERR,"Error writing density in reservoir/bias\n");
    //fflush(fp);
    fclose(fp);
  }

}

int FixReservoirBias::pack_forward_comm(int n, int *list, double *buf, int pbc_flag, int *pbc)
{
  int i,j,m;

  int *type = atom->type;

  m = 0;

  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = type[j];
  }

  return m;
}

/* ---------------------------------------------------------------------- */

void FixReservoirBias::unpack_forward_comm(int n, int first, double *buf)
{
  int i,m,last;

  int *type = atom->type;

  m = 0;
  last = first + n;

  for (i = first; i < last; i++) {
    type[i] = static_cast<int> (buf[m++]);
  }
}
