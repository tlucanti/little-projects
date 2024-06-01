#ifndef _GSIMULATION_HPP
#define _GSIMULATION_HPP

#include <random>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <stdlib.h>

#include <omp.h>

#include "Particle.hpp"

class GSimulation
{
public:
  GSimulation();
  ~GSimulation();

  void init();
  void set_number_of_particles(int N);
  void set_number_of_steps(int N);
  void start();

private:
  Particle *particles;

  int _npart;         // number of particles
  int _nsteps;        // number of integration steps
  real_type _tstep;   // time step of the simulation
  real_type _simtime; // total simulation time

  real_type _energy;  // energy of the system
  real_type _impulse; // impulse of the system

  void init_pos();
  void init_vel();
  void init_acc();
  void init_mass();

  inline void set_npart(const int &N) { _npart = N; }
  inline int get_npart() const { return _npart; }

  inline void set_simtime(const real_type &time) { _simtime = time; }
  inline real_type get_simtime() const { return _simtime; }

  inline void set_nsteps(const int &n) { _nsteps = n; }
  inline int get_nsteps() const { return _nsteps; }

  inline void init_tstep() { _tstep = _simtime / _nsteps; };
  inline real_type get_tstep() { return _tstep; };

  void print_header();
};

#endif
