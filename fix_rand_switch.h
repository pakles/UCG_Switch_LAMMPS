#ifdef FIX_CLASS

FixStyle(rand_switch,FixRandSwitch)

#else

#ifndef LMP_FIX_RAND_SWITCH_H
#define LMP_FIX_RAND_SWITCH_H

#include "fix.h"
#include <map>

namespace LAMMPS_NS {
  class FixRandSwitch : public Fix {
  public:
    FixRandSwitch(class LAMMPS *, int, char **);
    ~FixRandSwitch();
    int setmask();
    //    void consistencycheck();
    void allocate();
    void init();
    // void init_list(int, class NeighList *);
    void setup_pre_force(int);
    void setup_pre_force_respa(int, int);
    void pre_force(int);
    void pre_force_respa(int, int, int);

    //    void post_force_respa(int, int, int);
    //    void post_force(int);
    //    virtual void end_of_step();
    int pack_forward_comm(int, int *, double *, int, int *);
    void unpack_forward_comm(int, int, double *);
    double memory_usage();
    double compute_vector(int);


  private:
    int confirm_molecule(tagint); // checks molID state (returns 1 for ON and 0 for off)
    int switch_flag(int); // uses random to decided if this molID should switch state (returns 1 for YES)
    void read_file(char *);
    void attempt_switch(); // where all the MC switching happens
    void computecross(double *, double *, double *); // auxilliary functions cross product
    void normalize(double *); // auxilliary functions vector normalization
    void check_arrays(); // make sure all mol arrays are properly communicated and have the right information
    void gather_statistics(); //uses newly communicated mol arrays to gather MC statistics

    std::map<tagint,int> *hash; // hash map (key value) to keep track of mols

    int me, nprocs;
    int nlevels_respa;
    bool allocated;
    int pack_flag; //flag that controls which information is sent via comms
    //    int nlocal, nall; // number of atoms local to this processor, and including ghost atoms
    bigint ngroup; // number of atoms in group
    //int groupbit; //bitmask for group
    
    double probON; // probability of state 1 (ON)
    double probOFF; // probabilty of state 2 (OFF) = 1.0 - probON
    int nSwitchTypes; // number of atom types associated with state switching
    int *atomtypesON; // atom types associated with ON
    int *atomtypesOFF; // atom types associated with OFF
    int switchFreq; // number of timesteps between switching
    double nAttemptsTotal; // number of swap attempts which should equal numMols * (steps/switchFreq)
    double nSuccessTotal; // number of swap successes 
    double nAttemptsON; // number of swap attempts to ON
    double nAttemptsOFF; // number of swap attempts to OFF
    double nSuccessON; // number of swap successes to ON
    double nSuccessOFF; // number of swap successes to OFF

    class RanMars *random; //random number generator class

    int nmol; // number of molecules in group that are open to switching
    int maxmol; // maximum mol number of group to size array
    int *mol_restrict; // list of molecule ID tags that are open to switching
    tagint **mol_atoms; // 2D list of (mol ID), (internal atom type) = atom ID tag
    int *mol_state; //list of molecules current state
    int *mol_accept; //list of switching decisions for each molecule
    double ** cutsq;

    //    double *atom_state; //per-atom vector to output state of atom, i.e. working copy of vector_atom for the fix
    inline int sbmask(int j) {
      return j >> SBBITS & 3;
    }//similar to pair.h
  protected:
    int nmax;
  };
}
#endif
#endif
