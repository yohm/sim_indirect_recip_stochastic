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
    for (size_t i = 0; i < N; i++) { M[i].assign(N, Reputation::G); }
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
          if (g_prob_donor == 1.0) {
            M[obs][donor] = Reputation::G;
          }
          else if (g_prob_donor == 0.0) {
            M[obs][donor] = Reputation::B;
          }
          else {
            M[obs][donor] = (R01() < g_prob_donor) ? Reputation::G : Reputation::B;
          }
          if (mu_a > 0.0 && R01() < mu_a) {
            M[obs][donor] = FlipReputation(M[obs][donor]);
          }

          // update recipient's reputation
          double g_prob_recip = norms[obs].Rr.GProb(M[obs][recip], M[obs][donor], a_obs);
          if (g_prob_recip == 1.0) {
              M[obs][recip] = Reputation::G;
          }
          else if (g_prob_recip == 0.0) {
              M[obs][recip] = Reputation::B;
          }
          else {
              M[obs][recip] = (R01() < g_prob_recip) ? Reputation::G : Reputation::B;
          }
          if (mu_a > 0.0 && R01() < mu_a) {
            M[obs][recip] = FlipReputation(M[obs][recip]);
          }
        }
      }

      // count good
      for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < N; j++) {
          if (M[i][j] == Reputation::G) {
            good_count[i][j]++;
          }
        }
      }
      total_update++;

    }
  }

  using count_t = std::vector<std::vector<size_t>>;

  // number of C between i-donor and j-recipient
  count_t CoopCount() const { return coop_count; }

  // number of games between i-donor and j-recipient
  count_t GameCount() const { return game_count; }

  // system-wide cooperation level
  double SystemWideCooperationLevel() const {
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

  // donation levels of individuals
  // <probability of receiving benefit, probability of paying cost>
  // payoff
  std::vector<std::pair<double,double>> IndividualCooperationLevels() const {
    std::vector<std::pair<double,double>> coop_rates(N, {0.0, 0.0});
    for (size_t i = 0; i < N; i++) {
      double coop_total_out = 0.0, game_total_out = 0.0;
      for (size_t j = 0; j < N; j++) {
        if (i == j) {
          continue;
        }
        coop_total_out += coop_count[i][j];
        game_total_out += game_count[i][j];
      }
      double coop_total_in = 0.0, game_total_in = 0.0;
      for (size_t j = 0; j < N; j++) {
        if (i == j) {
          continue;
        }
        coop_total_in += coop_count[j][i];
        game_total_in += game_count[j][i];
      }
      coop_rates[i] = std::make_pair(coop_total_in / game_total_in, coop_total_out / game_total_out);
    }
    return coop_rates;
  }

  // norm-wise cooperation levels
  // return: vector c_levels
  //   c_levels[i][j] : cooperation level of i-th norm toward j-th norm
  std::vector<std::vector<double>> NormCooperationLevels() const {
    size_t n_norms = population.size();
    std::vector<std::vector<size_t>> coop_by_norm(n_norms), total_by_norm(n_norms);
    for (size_t i = 0; i < n_norms; i++) {
      coop_by_norm[i].resize(n_norms, 0.0);
        total_by_norm[i].resize(n_norms, 0.0);
    }
    for (size_t i = 0; i < N; i++) {
      for (size_t j = 0; j < N; j++) {
        size_t i_norm = norm_index[i];
        size_t j_norm = norm_index[j];
        coop_by_norm[i_norm][j_norm] += coop_count[i][j];
        total_by_norm[i_norm][j_norm] += game_count[i][j];
      }
    }
    std::vector<std::vector<double>> c_levels(n_norms);
    for (size_t i = 0; i < n_norms; i++) {
      c_levels[i].resize(n_norms, 0.0);
      for (size_t j = 0; j < n_norms; j++) {
        c_levels[i][j] = static_cast<double>(coop_by_norm[i][j]) / total_by_norm[i][j];
      }
    }
    return c_levels;
  }

  // norm-wise good count
  // return: vector average reputation
  //   c_levels[i][j] : average reputation of j-th norm from the viewpoint of i-th norm
  std::vector<std::vector<double>> NormAverageReputation() const {
    size_t n_norms = population.size();

    std::vector<std::vector<double>> avg_rep(n_norms);
    for (size_t i = 0; i < n_norms; i++) {
      avg_rep[i].resize(n_norms, 0.0);
    }

    for (size_t i = 0; i < N; i++) {
      for (size_t j = 0; j < N; j++) {
        size_t i_norm = norm_index[i];
        size_t j_norm = norm_index[j];
        avg_rep[i_norm][j_norm] += good_count[i][j];
      }
    }

    for (size_t i_norm = 0; i_norm < n_norms; i_norm++) {
      for (size_t j_norm = 0; j_norm < n_norms; j_norm++) {
        double size = population[i_norm].second * population[j_norm].second;
        avg_rep[i_norm][j_norm] /= (size * total_update);
      }
    }

    return avg_rep;
  }


  void ResetCounts() {
    coop_count.resize(N);
    game_count.resize(N);
    good_count.resize(N);
    for (size_t i = 0; i < N; i++) {
      coop_count[i].assign(N, 0);
      game_count[i].assign(N, 0);
      good_count[i].assign(N, 0);
    }
    total_update = 0;
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
  count_t good_count;  // number of good reputations
  size_t total_update; // number of time steps in total
  double R01() { return uni(rnd); }
};

class EvolutionaryPrivateRepGame {
public:
  EvolutionaryPrivateRepGame(size_t N, const std::vector<Norm>& norms) : N(N), norms(norms) {};

  const size_t N;
  const std::vector<Norm> norms;

  std::vector<std::vector<double>> FixationProbabilities(double benefit, double beta) const {
    size_t num_norms = norms.size();
    std::vector<std::vector<double>> rho(num_norms);
    for (size_t i = 0; i < num_norms; i++) {
      rho[i].resize(num_norms, 0.0);
    }
    for (size_t i = 0; i < num_norms; i++) {
      for (size_t j = i+1; j < num_norms; j++) {
        auto rho_ij_ji = FixationProbability(norms[i], norms[j], benefit, beta);
        rho[i][j] = rho_ij_ji.first;
        rho[j][i] = rho_ij_ji.second;
      }
    }
    return rho;
  }

  // first: fixation probability of resident j against resident i
  //        i.e., the probability to change from i to j
  // second: fixation probability of resident i against resident j
  //        i.e., the probability to change from j to i
  std::pair<double,double> FixationProbability(const Norm& norm_i, const Norm& norm_j, double benefit, double beta) const {
    std::vector<double> pi_i(N);  // pi_i[l]: payoff of resident i when l mutants exist
    std::vector<double> pi_j(N);  // pi_j[l]: payoff of mutant j when l mutants exist

    #pragma omp parallel for schedule(dynamic) shared(pi_i, pi_j)
    for (size_t l = 1; l < N; l++) {
      PrivateRepGame game({{norm_i, N-l}, {norm_j, l}}, 123456789ull);
      double q = 0.9, mu_percept = 0.05;
      game.Update(1e6, q, mu_percept);
      game.ResetCounts();
      game.Update(9e6, q, mu_percept);
      auto coop_levles = game.IndividualCooperationLevels();
      double payoff_i_total = 0.0;
      for (size_t k = 0; k < N-l; k++) {
        payoff_i_total += benefit * coop_levles[k].first - coop_levles[k].second;
      }
      pi_i[l] = payoff_i_total / (N-l);
      double payoff_j_total = 0.0;
      for (size_t k = N-l; k < N; k++) {
        payoff_j_total += benefit * coop_levles[k].first - coop_levles[k].second;
      }
      pi_j[l] = payoff_j_total / l;
      IC(l, pi_i[l], pi_j[l]);
    }
    std::cerr << "============" << std::endl;

    double rho_1_inv = 1.0;
    // p_ij = 1 / (1 + sum_{l' != 1}^{N-1}  prod_{l=1}^{l_prime} exp{-beta * (pi_j[l] - pi_i[l]) }
    for (size_t l_prime = 1; l_prime < N; l_prime++) {
      double prod = 1.0;
      for (size_t l = 1; l <= l_prime; l++) {
        prod *= exp(-beta * (pi_j[l] - pi_i[l]));
      }
      rho_1_inv += prod;
    }
    double rho_1 = 1.0 / rho_1_inv;

    double rho_2_inv = 1.0;
    for (size_t l_prime = 1; l_prime < N; l_prime++) {
      double prod = 1.0;
      for (size_t l = 1; l <= l_prime; l++) {
        prod *= exp(-beta * (pi_i[N-l] - pi_j[N-l]));
      }
      rho_2_inv += prod;
    }
    double rho_2 = 1.0 / rho_2_inv;
    return std::make_pair(rho_1, rho_2);
  }

  std::vector<double> EquilibriumPopulationLowMut(double benefit, double beta) const {
    // [TODO] implement me
    return {};
  }

};

int main() {

  // ActionRule ar({
  //                   {{G,G}, C},
  //                   {{G,B}, D},
  //                   {{B,G}, C},
  //                   {{B,B}, D},
  //               });

  // AssessmentRule Rd({
  //                       {{G,G,C}, 1.0}, {{G,G,D}, 0.0},
  //                       //{{G,B,C}, 0.5}, {{G,B,D}, 0.5},
  //                       {{G,B,C}, 1.0}, {{G,B,D}, 1.0},
  //                       {{B,G,C}, 1.0}, {{B,G,D}, 0.0},
  //                       {{B,B,C}, 1.0}, {{B,B,D}, 1.0},
  //                   });
  // AssessmentRule Rr = AssessmentRule::ImageScoring();
  // AssessmentRule Rr = AssessmentRule::KeepRecipient();

  // Norm norm(Rd, Rr, ar);
  // Norm norm = Norm::L1();
  // norm.Rr = AssessmentRule::ImageScoring();

  // PrivateRepGame priv_game( {{norm, 90}}, 123456789ull);
  // priv_game.Update(1000000, 0.9, 0.05);
  // // IC(priv_game.CoopCount(), priv_game.GameCount());
  // IC(priv_game.SystemWideCooperationLevel());

  IC(Norm::L1().Inspect());
  PrivateRepGame priv_game( {{Norm::L1(), 30}, {Norm::AllC(), 30}, {Norm::AllD(), 30}}, 123456789ull);
  priv_game.Update(1e6, 0.9, 0.05);
  priv_game.ResetCounts();
  priv_game.Update(1e6, 0.9, 0.05);
  IC( priv_game.NormCooperationLevels(), priv_game.NormAverageReputation() );
  // priv_game.PrintImage(std::cerr);

  // EvolutionaryPrivateRepGame evol(50, {Norm::L1(), Norm::AllC(), Norm::AllD()});
  // EvolutionaryPrivateRepGame evol(10, {Norm::L7(), Norm::L1()});
  // auto rhos = evol.FixationProbabilities(5.0, 1.0);
  // IC(rhos);

  return 0;
}