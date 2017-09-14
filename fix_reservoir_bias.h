/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(reservoir/bias,FixReservoirBias)

#else

#ifndef LMP_FIX_RESERVOIR_BIAS_H
#define LMP_FIX_RESERVOIR_BIAS_H

#include <stdio.h>
#include "fix.h"

namespace LAMMPS_NS {

class FixReservoirBias : public Fix {
 public:
  FixReservoirBias(class LAMMPS *, int, char **);
  ~FixReservoirBias();
  int setmask();
  void init();
  void setup(int);
  void post_force(int);
  double compute_scalar();
  double compute_vector(int);
  int pack_forward_comm(int, int *, double *, int, int *);
  void unpack_forward_comm(int, int, double *);

 private:
  double k; // force constant of reservoir barrier
  double w, w6, w6sq; // broadening of force (w in G(r,R)), and 6*w cutoff
  double n0; // number/area target for control region
  double nSL; // number/area threshold value to determine solid-liquid phase boundary
  double rCR; // radial thickness of control region
  double rSL; // radius of solid-liquid phase boundary
  //int nevery; // frequency of applied bias (default = 1), inherited from fix.h
  double nCR; // number/area within control region (calculated and communicated to all procs)
  double binwidth; // width of bin to calculate density and so forth
  double invbinwidth; // 1/binwidth;

  // arrays to store number/area info
  int nummaxbin; // for now, find the # of bins from domain size and add small buffer
  double *bincoord; // radial center of bin (assuming start is center of domain)
  double *numarea, *numarea_all; // number of atoms/area in each bin (only in xy direction)
  double indenter[6], indenter_all[6]; // stores energy and 3 force components from bias and rSL and nCR
  double center[3]; // center of simulation domain
  int indenter_flag, align_flag, align2_flag;

  int *mol_preferred_state; //list of molecules state based on radial position: >0 for assembly competent, <0 for assembly incapable
  int nmol; // number of molecules open to switching
  int maxmol; // maximum mol number to size array
  int nSwitchTypes; // number of atom types that are switchable
  int atomtypesON[4], atomtypesOFF[4]; // atom types that represent ON and OFF state (hard-coded for now)

  //FILE *fp;
  int me; //comm rank
  void calc_radial_density();
  void print_density();
  void apply_restraint(); //post_force?
  void activate_assembly_status(); //based on mol_state, see if types need to be switched. then update mol_state
  
};

}

#endif
#endif

