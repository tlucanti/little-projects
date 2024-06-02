#include "GSimulation.hpp"
#include "cpu_time.hpp"

std::mt19937 gen(time(nullptr));
std::uniform_real_distribution<real_type> unif_d(-1.0, 1.0);

GSimulation::GSimulation()
{
  std::cout << "===============================" << std::endl;
  std::cout << " Initialize Gravity Simulation" << std::endl;
  set_npart(10000);
  set_nsteps(10);
  set_simtime(1);
}

void GSimulation::set_number_of_particles(int N)
{
  set_npart(N);
}

void GSimulation::set_number_of_steps(int N)
{
  set_nsteps(N);
}

void GSimulation::init_pos()
{
  for (int i = 0; i < get_npart(); ++i)
  {
    particles[i].pos[0] = unif_d(gen);
    particles[i].pos[1] = unif_d(gen);
    particles[i].pos[2] = unif_d(gen);
  }
}

void GSimulation::init_vel()
{
  for (int i = 0; i < get_npart(); ++i)
  {
    particles[i].vel[0] = unif_d(gen);
    particles[i].vel[1] = unif_d(gen);
    particles[i].vel[2] = unif_d(gen);
  }
}

void GSimulation::init_acc()
{
  for (int i = 0; i < get_npart(); ++i)
  {
    particles[i].acc[0] = 0;
    particles[i].acc[1] = 0;
    particles[i].acc[2] = 0;
  }
}

void GSimulation::init_mass()
{
  real_type n = static_cast<real_type>(get_npart());
  for (int i = 0; i < get_npart(); ++i)
  {
    particles[i].mass = n * unif_d(gen);
  }
}

// prevents explosion in the case the particles are really close to each other
static constexpr double softeningSquared = 1e-9;
static constexpr double G = 6.67259e-11;

void sum_with_correction(double &sum, double &value_to_add, double &correction)
{
  double corrected = value_to_add - correction;
  double new_sum = sum + corrected;
  correction = (new_sum - sum) - corrected;
  sum = new_sum;
}

void compute_impulse(Particle *particles, int n, double sum_impulse[])
{
  double correction[] = {0, 0, 0};
  for (int i = 0; i < n; ++i)
  {
    for (int j = 0; j < 3; ++j)
    {
      double curr_impulse = particles[i].mass * particles[i].vel[j];
      sum_with_correction(sum_impulse[j], curr_impulse, correction[j]);
    }
  }
}

double compute_k_energy(Particle *particles, int num_parts)
{
  double energy = 0.;
  double correction = 0.;
  for (int i = 0; i < num_parts; ++i)
  {
    double curr_energy = particles[i].mass * (particles[i].vel[0] * particles[i].vel[0] +
                                              particles[i].vel[1] * particles[i].vel[1] +
                                              particles[i].vel[2] * particles[i].vel[2]);
    sum_with_correction(energy, curr_energy, correction);
  }
  return energy / 2;
}

double compute_p_energy(Particle *particles, int num_parts)
{
  double p_energy = 0.;
  double correction = 0.;
  for (int i = 0; i < num_parts; ++i)
  {
    for (int j = 0; j < num_parts; ++j)
    {
      if (i == j)
        continue;
      double dx = particles[j].pos[0] - particles[i].pos[0];
      double dy = particles[j].pos[1] - particles[i].pos[1];
      double dz = particles[j].pos[2] - particles[i].pos[2];

      double distanceSqr = dx * dx + dy * dy + dz * dz + softeningSquared;
      double distanceInv = 1.0 / sqrt(distanceSqr);
      double curr_energy = -G * particles[j].mass * distanceInv * particles[i].mass;
      sum_with_correction(p_energy, curr_energy, correction);
    }
  }
  return p_energy / 2;
}

void partial_step(double dst[3], double a[3], double b[3], double k)
{
	dst[0] = a[0] + b[0] * k;
	dst[1] = a[1] + b[1] * k;
	dst[2] = a[2] + b[2] * k;
}

void computeAcc(Particle *buff, int part_num, double time_step)
{
  for (int i = 0; i < part_num; i++) // update acceleration
  {
    buff[i].acc[0] = 0;
    buff[i].acc[1] = 0;
    buff[i].acc[2] = 0;

    for (int j = 0; j < part_num; j++)
    {
		if (i == j) {
			continue;
		}

	    	Particle *target_body = &buff[i];
		Particle *external_body = &buff[j];

		double k1[3], k2[3], k3[3], k4[3], tmp_vel[3], tmp_loc[3];

                double r = pow((target_body->pos[0] - external_body->pos[0]), 2) +
			   pow((target_body->pos[1] - external_body->pos[1]), 2) +
			   pow((target_body->pos[2] - external_body->pos[2]), 2);
                r = sqrt(r);
                double tmp = G * external_body->mass / pow(r, 3);

                // k1 - regular Euler acceleration
                k1[0] = tmp * (external_body->pos[0] - target_body->pos[0]);
                k1[1] = tmp * (external_body->pos[1] - target_body->pos[1]);
                k1[2] = tmp * (external_body->pos[2] - target_body->pos[2]);

                // k2 - acceleration 0.5 timesteps in the future based on k1 acceleration value
                partial_step(tmp_vel, target_body->vel, k1, 0.5);
                partial_step(tmp_loc, target_body->pos, tmp_vel, 0.5 * time_step);
                k2[0] = (external_body->pos[0] - tmp_loc[0]) * tmp;
                k2[1] = (external_body->pos[1] - tmp_loc[1]) * tmp;
                k2[2] = (external_body->pos[2] - tmp_loc[2]) * tmp;

                // k3 acceleration 0.5 timesteps in the future using k2 acceleration
                partial_step(tmp_vel, target_body->vel, k2, 0.5);
                partial_step(tmp_loc, target_body->pos, tmp_vel, 0.5 * time_step);
                k3[0] = (external_body->pos[0] - tmp_loc[0]) * tmp;
                k3[1] = (external_body->pos[1] - tmp_loc[1]) * tmp;
                k3[2] = (external_body->pos[2] - tmp_loc[2]) * tmp;

                // k4 - location 1 timestep in the future using k3 acceleration
                partial_step(tmp_vel, target_body->vel, k3, 1);
                partial_step(tmp_loc, target_body->pos, tmp_vel, time_step);
                k4[0] = (external_body->pos[0] - tmp_loc[0]) * tmp;
                k4[1] = (external_body->pos[1] - tmp_loc[1]) * tmp;
                k4[2] = (external_body->pos[2] - tmp_loc[2]) * tmp;

                target_body->acc[0] += (k1[0] + k2[0] * 2 + k3[0] * 2 + k4[0]) / 6;
                target_body->acc[1] += (k1[1] + k2[1] * 2 + k3[1] * 2 + k4[1]) / 6;
                target_body->acc[2] += (k1[2] + k2[2] * 2 + k3[2] * 2 + k4[2]) / 6;
    }
  }
}

void update_pos(Particle *dst, const Particle *src_1, const Particle *src_2, double coef, int part_num)
{
  for (int i = 0; i < part_num; ++i)
  {
    dst[i].pos[0] = src_1[i].pos[0] + src_2[i].vel[0] * coef;
    dst[i].pos[1] = src_1[i].pos[1] + src_2[i].vel[1] * coef;
    dst[i].pos[2] = src_1[i].pos[2] + src_2[i].vel[2] * coef;

    dst[i].vel[0] = src_1[i].vel[0] + src_2[i].acc[0] * coef;
    dst[i].vel[1] = src_1[i].vel[1] + src_2[i].acc[1] * coef;
    dst[i].vel[2] = src_1[i].vel[2] + src_2[i].acc[2] * coef;
  }
}

void GSimulation::start()
{
  init_tstep();
  real_type energy_k, energy_p;
  real_type dt = get_tstep();
  int n = get_npart();

  particles = new Particle[n];

  init_pos();
  init_vel();
  init_acc();
  init_mass();

  energy_k = compute_k_energy(particles, n);
  energy_p = compute_p_energy(particles, n);
  _energy = energy_k + energy_p;
  double impulse[] = {0, 0, 0};
  compute_impulse(particles, n, impulse);
  _impulse = sqrt(pow(impulse[0], 2) + pow(impulse[1], 2) + pow(impulse[2], 2));

  std::cout << "Initial system energy k: " << energy_k << " p:" << energy_p << " Sum: " << _energy << " Impulse: " << _impulse << std::endl;

  print_header();

  double _totTime = 0.;

  CPUTime time;
  double ts0 = 0;
  double ts1 = 0;

  const double t0 = time.start();
  for (int s = 1; s <= get_nsteps(); ++s)
  {
    ts0 += time.start();

    // Simple Euler Method
    computeAcc(particles, n, dt / 2);
    update_pos(particles, particles, particles, dt / 2, n);

    energy_k = compute_k_energy(particles, n);
    energy_p = compute_p_energy(particles, n);

    double curr_energy = energy_k + energy_p;
    impulse[0] = impulse[1] = impulse[2] = 0;
    compute_impulse(particles, n, impulse);

    double curr_impulse = sqrt(impulse[0] * impulse[0] + impulse[1] * impulse[1] + impulse[2] * impulse[2]);

    ts1 += time.stop();
    // if (!(s % get_sfreq()))
    if (1)
    {
      std::cout << " "
                << std::left << std::setw(8) << s
                << std::left << std::setprecision(5) << std::setw(8) << s * get_tstep()
                << std::left << std::setprecision(9) << std::setw(16) << fabs(100 * (curr_energy - _energy) / _energy)
                << std::left << std::setprecision(9) << std::setw(16) << fabs(100 * (curr_impulse - _impulse) / _impulse)
                << std::left << std::setprecision(5) << std::setw(16) << (ts1 - ts0)
                << std::endl;
      ts0 = 0;
      ts1 = 0;
    }
    // _energy = curr_energy;
    // _impulse = curr_impulse;
  } // end of the time step loop

  const double t1 = time.stop();
  _totTime = (t1 - t0);

  std::cout << std::endl;
  std::cout << "# Total Time (s)     : " << _totTime << std::endl;
  std::cout << "===============================" << std::endl;
}

void GSimulation ::print_header()
{

  std::cout << " nPart = " << get_npart() << "; "
            << "nSteps = " << get_nsteps() << "; "
            << "dt = " << get_tstep() << std::endl;

  std::cout << "------------------------------------------------" << std::endl;
  std::cout << " "
            << std::left << std::setw(8) << "s"
            << std::left << std::setw(8) << "dt"
            << std::left << std::setw(16) << "s_energy"
            << std::left << std::setw(16) << "impulse"
            << std::left << std::setw(16) << "time (s)"
            << std::endl;
  std::cout << "------------------------------------------------" << std::endl;
}

GSimulation ::~GSimulation()
{
}
