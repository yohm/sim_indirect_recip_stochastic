#include <iostream>
#include <cassert>
#include <icecream.hpp>
#include <vector>
#include <random>
#include <Eigen/Dense>
#include "Norm.hpp"
#include "PrivRepGame.hpp"


using Reputation::G, Reputation::B, Action::C, Action::D;


class EvolutionaryPrivateRepGame {
public:
  EvolutionaryPrivateRepGame(size_t N, const std::vector<Norm>& norms) : N(N), norms(norms) {};

  const size_t N;
  const std::vector<Norm> norms;

  // rho[i][j] = fixation probability of a j-mutant into i-resident community
  std::vector<std::vector<double>> FixationProbabilities(double benefit, double beta) const {
    size_t num_norms = norms.size();
    std::vector<std::vector<double>> rho(num_norms, std::vector<double>(num_norms, 0.0));

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
      game.Update(1e6, q, mu_percept, false);
      game.ResetCounts();
      game.Update(1e6, q, mu_percept, false);
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

  // arg: transition probability matrix p
  //      p[i][j] : fixation probability of a j-mutant into i-resident
  // return: stationary distribution vector
  std::vector<double> EquilibriumPopulationLowMut(const std::vector<std::vector<double>>& fixation_probs) const {
    size_t num_norms = fixation_probs.size();

    // construct transition matrix
    Eigen::MatrixXd T(num_norms, num_norms);

    // std::vector<std::vector<double>> T(num_norms, std::vector<double>(num_norms, 0.0));
    for (size_t i = 0; i < num_norms; i++) {
      double sum = 0.0;
      for (size_t j = 0; j < num_norms; j++) {
        if (i == j) continue;
        T(j,i) = fixation_probs[i][j] / (1.0 - num_norms);
        sum += T(j, i);
      }
      T(i,i) = - sum;  // T(i,i) = 1.0 - sum; but we subtract I to calculate stationary distribution
    }
    for (size_t i = 0; i < num_norms; i++) {  // normalization condition
      T(num_norms-1, i) += 1.0;
    }

    Eigen::VectorXd b(num_norms);
    for (int i = 0; i < num_norms-1; i++) { b(i) = 0.0; }
    b(num_norms-1) = 1.0;

    Eigen::VectorXd x = T.colPivHouseholderQr().solve(b);

    std::vector<double> ans(num_norms, 0.0);
    for (int i = 0; i < num_norms; i++) {
      ans[i] = x(i);
    }
    return ans;
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

  /*
  IC(Norm::L1().Inspect());
  PrivateRepGame priv_game( {{Norm::L1(), 30}, {Norm::AllC(), 30}, {Norm::AllD(), 30}}, 123456789ull);
  priv_game.Update(1e6, 0.9, 0.05);
  priv_game.ResetCounts();
  priv_game.Update(1e6, 0.9, 0.05);
  IC( priv_game.NormCooperationLevels(), priv_game.NormAverageReputation() );
  // priv_game.PrintImage(std::cerr);
   */

  /*
  Norm norm = Norm::L3();
  // norm.Rd.SetGProb(G, B, D, 0.5);
  // norm.Rd.SetGProb(G, B, C, 0.5);
  norm.Rr = AssessmentRule::ImageScoring();
  // norm.Rr = AssessmentRule::KeepRecipient();

  PrivateRepGame priv_game( {{norm, 50}}, 123456789ull);
  priv_game.Update(1e6, 0.9, 0.05, false);
  priv_game.ResetCounts();
  priv_game.Update(1e6, 0.9, 0.05, false);
  IC( priv_game.NormCooperationLevels() );

  EvolutionaryPrivateRepGame evol(50, {norm, Norm::AllC(), Norm::AllD()});
  auto rhos = evol.FixationProbabilities(5.0, 1.0);
  IC(rhos);
  auto eq = evol.EquilibriumPopulationLowMut(rhos);
  IC(eq);
   */

  // random norm
  ActionRule ar({ {{G,G}, 0.5}, {{G,B}, 0.5}, {{B,G}, 0.5}, {{B,B}, 0.5}, });
  AssessmentRule Rd({ {{G,G,C}, 0.5}, {{G,G,D}, 0.5}, {{G,B,C}, 0.5}, {{G,B,D}, 0.5}, {{B,G,C}, 0.5}, {{B,G,D}, 0.5}, {{B,B,C}, 0.5}, {{B,B,D}, 0.5}, });
  AssessmentRule Rr({ {{G,G,C}, 0.5}, {{G,G,D}, 0.5}, {{G,B,C}, 0.5}, {{G,B,D}, 0.5}, {{B,G,C}, 0.5}, {{B,G,D}, 0.5}, {{B,B,C}, 0.5}, {{B,B,D}, 0.5}, });
  Norm norm(Rd, Rr, ar);
  PrivateRepGame priv_game( {{norm, 50}}, 123456789ull);
  priv_game.Update(1e6, 0.9, 0.05, false);
  priv_game.ResetCounts();
  priv_game.Update(1e6, 0.9, 0.05, false);
  IC( priv_game.NormCooperationLevels() );
  priv_game.PrintImage(std::cout);
  return 0;
}