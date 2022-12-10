#include <iostream>
#include <cassert>
#include <icecream.hpp>
#include <vector>
#include <random>
#include "Norm.hpp"
#include "Game.hpp"


using Reputation::G, Reputation::B, Action::C, Action::D;

class PrivateRepGame {
public:
  using population_t = std::vector<std::pair<Norm, size_t>>;  // vector of StrategyID & its size

  PrivateRepGame(const population_t& pop, uint64_t _seed) : population(pop), rnd(_seed), uni(0.0, 1.0) {
    for (size_t i = 0; i < population.size(); i++) {
      auto kv = population[i];
      for (size_t ii = 0; ii < kv.second; ii++) {
        norms.emplace_back(kv.first);
        norm_index.emplace_back(i);
      }
    }
    N = norms.size();
    M.resize(N);
    for (size_t i = 0; i < N; i++) { M[i].resize(N, Reputation::G); }
    ResetCounts();
  }

  // t_max : number of steps
  // q : observation probability
  // epsilon :
  void Update(size_t t_max, double q, double mu_percept, double mu_e = 0.0, double mu_a = 0.0) {
    for (size_t t = 0; t < t_max; t++) {
      // randomly choose donor & recipient
      size_t donor = static_cast<size_t>(R01() * N);
      size_t recip = (donor + static_cast<size_t>(R01() * (N-1)) + 1) % N;
      assert(donor != recip);

      double c_prob = norms[donor].P.CProb(M[donor][donor], M[donor][recip]);
      Action A = (R01() < c_prob) ? Action::C : Action::D;
      // implementation error
      if(mu_e > 0.0 && R01() < mu_e) { A = Action::D; }

      if (A == Action::C) {
        coop_count[donor][recip]++;
      }
      game_count[donor][recip]++;

      // updating the images from observers' viewpoint
      for (size_t obs = 0; obs < N; obs++) {
        if (obs == donor || obs == recip || R01() < q) {  // observe with probability q
          Action a_obs = A;
          if (mu_percept > 0.0 && R01() < mu_percept) {
            a_obs = FlipAction(A);
          }

          // update donor's reputation
          double g_prob_donor = norms[obs].Rd.GProb(M[obs][donor], M[obs][recip], a_obs);
          if (R01() < g_prob_donor) {
            M[obs][donor] = Reputation::G;
          } else {
            M[obs][donor] = Reputation::B;
          }
          if (mu_a > 0.0 && R01() < mu_a) {
            M[obs][donor] = FlipReputation(M[obs][donor]);
          }

          // update recipient's reputation
          double g_prob_recip = norms[obs].Rr.GProb(M[obs][recip], M[obs][donor], a_obs);
          if (R01() < g_prob_recip) {
              M[obs][recip] = Reputation::G;
          } else {
              M[obs][recip] = Reputation::B;
          }
          if (mu_a > 0.0 && R01() < mu_a) {
            M[obs][recip] = FlipReputation(M[obs][recip]);
          }
        }
      }
    }
  }

  using count_t = std::vector<std::vector<size_t>>;

  // number of C between i-donor and j-recipient
  count_t CoopCount() const { return coop_count; }

  // number of games between i-donor and j-recipient
  count_t GameCount() const { return game_count; }

  // system-wide cooperation level
  double CooperationLevel() const {
    double coop = 0.0;
    double total = 0.0;
    for (size_t i = 0; i < N; i++) {
      for (size_t j = 0; j < N; j++) {
        coop += coop_count[i][j];
        total += game_count[i][j];
      }
    }
    return coop / total;
  }

  // reset coop_count and game_count
  void ResetCounts() {
    coop_count.resize(N);
    game_count.resize(N);
    for (size_t i = 0; i < N; i++) {
      coop_count[i].resize(N, 0);
      game_count[i].resize(N, 0);
    }
  }

  // print image matrix
  void PrintImage(std::ostream& out) const {
    for (size_t i = 0; i < N; i++) {
      for (size_t j = 0; j < N; j++) {
        out << M[i][j] << " ";
      }
      out << std::endl;
    }
  }

private:
  const population_t population;
  size_t N;
  std::mt19937_64 rnd;
  std::uniform_real_distribution<double> uni;
  std::vector<Norm> norms;
  std::vector<size_t> norm_index;  // [0,0,...,0, 1,1,....,1, 2,2,......,2]
  std::vector<std::vector<Reputation> > M;
  count_t coop_count;  // number of C between i-donor and j-recipient
  count_t game_count;  // number of games between i-donor and j-recipient
  double R01() { return uni(rnd); }
};


int main() {

  ActionRule ar({
                    {{G,G}, C},
                    {{G,B}, D},
                    {{B,G}, C},
                    {{B,B}, D},
                });

  AssessmentRule Rd({
                        {{G,G,C}, 1.0}, {{G,G,D}, 0.0},
                        {{G,B,C}, 0.5}, {{G,B,D}, 0.5},
                        {{B,G,C}, 1.0}, {{B,G,D}, 0.0},
                        {{B,B,C}, 1.0}, {{B,B,D}, 1.0},
                    });
  AssessmentRule Rr({
                        {{G,G,C}, 1.0}, {{G,G,D}, 0.0},
                        {{G,B,C}, 1.0}, {{G,B,D}, 0.0},
                        {{B,G,C}, 1.0}, {{B,G,D}, 0.0},
                        {{B,B,C}, 1.0}, {{B,B,D}, 0.0},
                    });

  Norm norm(Rd, Rr, ar);
  Norm l3 = Norm::L3();

  PrivateRepGame priv_game( {{l3, 90}}, 123456789ull);
  priv_game.Update(1000000, 0.9, 0.05);
  // IC(priv_game.CoopCount(), priv_game.GameCount());
  IC(priv_game.CooperationLevel());

  // priv_game.PrintImage(std::cerr);

  return 0;
}