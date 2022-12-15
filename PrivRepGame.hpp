#include <iostream>
#include <vector>
#include <random>
#include "Norm.hpp"

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
    M.assign(N, std::vector<Reputation>(N, Reputation::G));
    ResetCounts();
  }

  // t_max : number of steps
  // q : observation probability
  // epsilon :
  void Update(size_t t_max, double q, double mu_percept, bool count_good = true) {
    for (size_t t = 0; t < t_max; t++) {
      // randomly choose donor & recipient
      size_t donor = static_cast<size_t>(R01() * N);
      size_t recip = (donor + static_cast<size_t>(R01() * (N-1)) + 1) % N;
      assert(donor != recip);

      double c_prob = norms[donor].P.CProb(M[donor][donor], M[donor][recip]);
      Action A = (c_prob == 1.0 || R01() < c_prob) ? Action::C : Action::D;

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

          // update recipient's reputation
          double g_prob_recip = norms[obs].Rr.GProb(M[obs][donor], M[obs][recip], a_obs);
          if (g_prob_recip == 1.0) {
            M[obs][recip] = Reputation::G;
          }
          else if (g_prob_recip == 0.0) {
            M[obs][recip] = Reputation::B;
          }
          else {
            M[obs][recip] = (R01() < g_prob_recip) ? Reputation::G : Reputation::B;
          }
        }
      }

      // count good
      if (count_good) {
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
    std::vector<std::vector<size_t>> coop_by_norm(n_norms, std::vector<size_t>(n_norms, 0) );
    std::vector<std::vector<size_t>> total_by_norm(n_norms, std::vector<size_t>(n_norms, 0) );
    for (size_t i = 0; i < N; i++) {
      for (size_t j = 0; j < N; j++) {
        size_t i_norm = norm_index[i];
        size_t j_norm = norm_index[j];
        coop_by_norm[i_norm][j_norm] += coop_count[i][j];
        total_by_norm[i_norm][j_norm] += game_count[i][j];
      }
    }
    std::vector<std::vector<double>> c_levels(n_norms, std::vector<double>(n_norms, 0.0));
    for (size_t i_norm = 0; i_norm < n_norms; i_norm++) {
      for (size_t j_norm = 0; j_norm < n_norms; j_norm++) {
        c_levels[i_norm][j_norm] = static_cast<double>(coop_by_norm[i_norm][j_norm]) / static_cast<double>(total_by_norm[i_norm][j_norm]);
      }
    }
    return c_levels;
  }

  // norm-wise good count
  // return: vector average reputation
  //   c_levels[i][j] : average reputation of j-th norm from the viewpoint of i-th norm
  std::vector<std::vector<double>> NormAverageReputation() const {
    size_t n_norms = population.size();
    std::vector<std::vector<double>> avg_rep(n_norms, std::vector<double>(n_norms, 0.0));

    for (size_t i = 0; i < N; i++) {
      for (size_t j = 0; j < N; j++) {
        size_t i_norm = norm_index[i];
        size_t j_norm = norm_index[j];
        avg_rep[i_norm][j_norm] += good_count[i][j];
      }
    }

    for (size_t i_norm = 0; i_norm < n_norms; i_norm++) {
      for (size_t j_norm = 0; j_norm < n_norms; j_norm++) {
        size_t c = population[i_norm].second * population[j_norm].second * total_update;
        avg_rep[i_norm][j_norm] /= static_cast<double>(c);
      }
    }

    return avg_rep;
  }


  void ResetCounts() {
    coop_count.assign(N, std::vector<size_t>(N, 0));
    game_count.assign(N, std::vector<size_t>(N, 0));
    good_count.assign(N, std::vector<size_t>(N, 0));
    total_update = 0;
  }

  // print image matrix
  void PrintImage(std::ostream& out) const {
    for (size_t i = 0; i < N; i++) {
      for (size_t j = 0; j < N; j++) {
        out << ((M[i][j]==Reputation::G)?'.':'x');
      }
      out << "\n";
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

