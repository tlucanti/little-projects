#include <iostream>

#include "GSimulation.hpp"

int main(int argc, char **argv)
{
  int N;     // number of particles
  int nstep; // number ot integration steps

  if (argc == 3)
  {
    GSimulation sim;
    N = atoi(argv[1]);
    sim.set_number_of_particles(N);

    nstep = atoi(argv[2]);
    sim.set_number_of_steps(nstep);

    sim.start();
  }
  else
  {
    std::cout << "Use: n_body_simulation.exe PARTICLES_NUM STEPS_NUM\n";
  }

  return 0;
}
