#include "GSimulation.hpp"
#include "cpu_time.hpp"

double random_double(void)
{
	return (double)rand() / RAND_MAX * 2 - 1;
}

Point random_point(void)
{
	return { random_double(), random_double(), random_double() };
}

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
    particles.at(i).pos = random_point();
  }
}

void GSimulation::init_vel()
{
  for (int i = 0; i < get_npart(); ++i)
  {
    particles.at(i).vel = random_point();
  }
}

void GSimulation::init_acc()
{
  for (int i = 0; i < get_npart(); ++i)
  {
    particles.at(i).acc = { 0, 0, 0 };
  }
}

void GSimulation::init_mass()
{
  for (int i = 0; i < get_npart(); ++i)
  {
    particles.at(i).mass = std::abs(random_double()) * 1e5;
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

void compute_impulse(std::vector<Particle> &particles, int n, Point &sum_impulse)
{
  Point correction = {0, 0, 0};
  for (int i = 0; i < n; ++i)
  {
    for (int j = 0; j < 3; ++j)
    {
      double curr_impulse = particles.at(i).mass * particles.at(i).vel[j];
      sum_with_correction(sum_impulse[j], curr_impulse, correction[j]);
    }
  }
}

double compute_k_energy(std::vector<Particle> &particles, int num_parts)
{
  double energy = 0.;
  double correction = 0.;
  for (int i = 0; i < num_parts; ++i)
  {
    double curr_energy = particles.at(i).mass * (particles.at(i).vel[0] * particles.at(i).vel[0] +
                                              particles.at(i).vel[1] * particles.at(i).vel[1] +
                                              particles.at(i).vel[2] * particles.at(i).vel[2]);
    sum_with_correction(energy, curr_energy, correction);
  }
  return energy / 2;
}

double compute_p_energy(std::vector<Particle> &particles, int num_parts)
{
  double p_energy = 0.;
  double correction = 0.;
  for (int i = 0; i < num_parts; ++i)
  {
    for (int j = 0; j < num_parts; ++j)
    {
      if (i == j)
        continue;
      double dx = particles.at(j).pos[0] - particles.at(i).pos[0];
      double dy = particles.at(j).pos[1] - particles.at(i).pos[1];
      double dz = particles.at(j).pos[2] - particles.at(i).pos[2];

      double distanceSqr = dx * dx + dy * dy + dz * dz + softeningSquared;
      double distanceInv = 1.0 / sqrt(distanceSqr);
      double curr_energy = -G * particles.at(j).mass * distanceInv * particles[i].mass;
      sum_with_correction(p_energy, curr_energy, correction);
    }
  }
  return p_energy / 2;
}

#define MIN_SIMULATED_DIST 1e-6

void computeAccOnce(std::vector<Particle> &bodies, size_t cur, double time_step)
{
	Point acc = { 0, 0, 0 };
	Point tmp_vel;
	Point tmp_pos;

	for (size_t other = 0; other < bodies.size(); other++) {
		if (cur == other) {
			continue;
		}

		Point dist = bodies.at(other).pos - bodies.at(cur).pos;
		double r = dist.abs();
		r = std::pow(std::sqrt(r), 3);
		if (r < MIN_SIMULATED_DIST) {
			r = MIN_SIMULATED_DIST;
		}
		double norm = G * bodies.at(other).mass / r;
		Point k1 = (bodies.at(other).pos - bodies.at(cur).pos) * norm;

		tmp_vel = bodies.at(cur).vel + k1 * 0.5 * time_step;
		tmp_pos = bodies.at(cur).pos + tmp_vel * 0.5 * time_step;
		Point k2 = (bodies.at(other).pos - tmp_pos) * norm;

		tmp_vel = bodies.at(cur).vel + k2 * 0.5 * time_step;
		tmp_pos = bodies.at(cur).pos + tmp_vel * 0.5 * time_step;
		Point k3 = (bodies.at(other).pos - tmp_pos) * norm;

		tmp_vel = bodies.at(cur).vel + k3 * time_step;
		tmp_pos = bodies.at(cur).pos + tmp_vel * time_step;
		Point k4 = (bodies.at(other).pos - tmp_pos) * norm;

		acc = acc + (k1 + k2 * 2 + k3 * 3 + k4) / (double)6 * time_step;
	}

	bodies.at(cur).acc = acc;
}

void computeAcc(std::vector<Particle> &buff, double time_step)
{
  for (size_t i = 0; i < buff.size(); i++) // update acceleration
  {
    buff.at(i).acc = { 0, 0, 0 };
    computeAccOnce(buff, i, time_step);
  }
}

void update_pos(std::vector<Particle> &dst, std::vector<Particle> &src_1, std::vector<Particle> &src_2, double coef, int part_num)
{
  for (int i = 0; i < part_num; ++i)
  {
    dst.at(i).pos[0] = src_1.at(i).pos[0] + src_2.at(i).vel[0] * coef;
    dst.at(i).pos[1] = src_1.at(i).pos[1] + src_2.at(i).vel[1] * coef;
    dst.at(i).pos[2] = src_1.at(i).pos[2] + src_2.at(i).vel[2] * coef;

    dst.at(i).vel[0] = src_1.at(i).vel[0] + src_2.at(i).acc[0] * coef;
    dst.at(i).vel[1] = src_1.at(i).vel[1] + src_2.at(i).acc[1] * coef;
    dst.at(i).vel[2] = src_1.at(i).vel[2] + src_2.at(i).acc[2] * coef;
  }
}

void GSimulation::start()
{
  init_tstep();
  real_type energy_k, energy_p;
  int n = get_npart();

  particles = std::vector<Particle>(n);

  init_pos();
  init_vel();
  init_acc();
  init_mass();

  energy_k = compute_k_energy(particles, n);
  energy_p = compute_p_energy(particles, n);
  _energy = energy_k + energy_p;
  Point impulse = {0, 0, 0};
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

    computeAcc(particles, get_tstep());
    update_pos(particles, particles, particles, get_tstep(), n);

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
