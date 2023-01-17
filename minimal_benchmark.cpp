#include <iostream>
#include <chrono>
#include <random>
#include "Norm.hpp"


const size_t N = 50;
std::mt19937_64 rnd(123456789ull);
std::vector<Norm> norms;
std::vector<std::vector<Reputation> > M;
std::uniform_real_distribution<double> uni(0.0, 1.0);
double R01() { return uni(rnd); }

void Initialize(const Norm& norm) {
  norms.emplace_back(norm);
  M.assign(N, std::vector<Reputation>(N, Reputation::G));
}

void Update(size_t t_max, double q, double mu_percept) {
  for (size_t t = 0; t < t_max; t++) {
    // randomly choose donor & recipient
    size_t donor = static_cast<size_t>(R01() * N);
    size_t recip = (donor + static_cast<size_t>(R01() * (N-1)) + 1) % N;

    double c_prob = norms[donor].P.CProb(M[donor][donor], M[donor][recip]);
    Action A = (c_prob == 1.0 || R01() < c_prob) ? Action::C : Action::D;

    // updating the images from observers' viewpoint
    for (size_t obs = 0; obs < N; obs++) {
      if (obs == donor || obs == recip || R01() < q) {  // observe with probability q
        Action a_obs = A;
        if (mu_percept > 0.0 && R01() < mu_percept) {
          a_obs = FlipAction(A);
        }

        // update donor's reputation
        double g_prob_donor = norms[obs].Rd.GProb(M[obs][donor], M[obs][recip], a_obs);
        M[obs][donor] = (R01() < g_prob_donor) ? Reputation::G : Reputation::B;

        // update recipient's reputation
        double g_prob_recip = norms[obs].Rr.GProb(M[obs][donor], M[obs][recip], a_obs);
        M[obs][recip] = (R01() < g_prob_recip) ? Reputation::G : Reputation::B;
      }
    }
  }
}

void Benchmark() {
  std::cout << "Benchmarking..." << std::endl;
  Initialize(Norm::Random());
  auto start = std::chrono::system_clock::now();
  Update(1e5, 0.9, 0.05);
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "elapsed time: " << elapsed_seconds.count() << "s" << std::endl;
}

int main(int argc, char *argv[]) {

  Benchmark();

  return 0;
}