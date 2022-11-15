#ifndef GAME_HPP
#define GAME_HPP

#include <iostream>
#include <sstream>
#include <cmath>
#include <functional>
#include <iomanip>
#include <icecream.hpp>
#include "Norm.hpp"

class Game {
public:
  Game(double mu_e, double mu_a_donor, double mu_a_recip, const Norm& norm) :
  mu_e(mu_e), mu_a_donor(mu_a_donor), mu_a_recip(mu_a_recip), norm(norm), r_norm(norm.RescaleWithError(mu_e, mu_a_donor, mu_a_recip)) {
    h_star = ResidentEqReputation();
    pc_res_res = h_star * h_star * r_norm.CProb(Reputation::G, Reputation::G)
        + h_star * (1.0-h_star) * r_norm.CProb(Reputation::G, Reputation::B)
        + (1.0-h_star) * h_star * r_norm.CProb(Reputation::B, Reputation::G)
        + (1.0-h_star) * (1.0-h_star) * r_norm.CProb(Reputation::B, Reputation::B);
  }
  std::string Inspect() const {
    std::stringstream ss;
    ss << "Game:" << std::endl
       << "(mu_e, mu_a_donor, mu_a_recip): (" << mu_e << ", " << mu_a_donor << ", " << mu_a_recip << ")" << std::endl
       << "(Norm): " << norm.Inspect() << "" << std::endl
       << "(c_prob,h*): " << pc_res_res << ' ' << h_star << std::endl;
    return ss.str();
  }
  const double mu_e, mu_a_donor, mu_a_recip;
  double h_star; // equilibrium reputation of resident species
  double pc_res_res;  // cooperation probability of resident species
  const Norm norm;
  const Norm r_norm;  // norm that takes into account error rates (mu_e, mu_a)
  double ResidentEqReputation() const {
    // dynamcis of h(t)
    // dot{h(t)} =
    //   h^2    { R1(G,G) + R2(G,G) }
    // + h(1-h) { R1(G,B) + R2(G,B) + R1(B,G) + R2(B,G) }
    // + (1-h)^2{ R1(B,B) + R2(B,B) }
    // -2h(t)

    using Reputation::G, Reputation::B, Action::C, Action::D;
    double Rd_BB = r_norm.CProb(B, B) * r_norm.GProbDonor(B, B, C) + (1.0 - r_norm.CProb(B, B)) * r_norm.GProbDonor(B, B, D);
    double Rd_BG = r_norm.CProb(B, G) * r_norm.GProbDonor(B, G, C) + (1.0 - r_norm.CProb(B, G)) * r_norm.GProbDonor(B, G, D);
    double Rd_GB = r_norm.CProb(G, B) * r_norm.GProbDonor(G, B, C) + (1.0 - r_norm.CProb(G, B)) * r_norm.GProbDonor(G, B, D);
    double Rd_GG = r_norm.CProb(G, G) * r_norm.GProbDonor(G, G, C) + (1.0 - r_norm.CProb(G, G)) * r_norm.GProbDonor(G, G, D);

    double Rr_BB = r_norm.CProb(B, B) * r_norm.GProbRecip(B, B, C) + (1.0 - r_norm.CProb(B, B)) * r_norm.GProbRecip(B, B, D);
    double Rr_BG = r_norm.CProb(B, G) * r_norm.GProbRecip(B, G, C) + (1.0 - r_norm.CProb(B, G)) * r_norm.GProbRecip(B, G, D);
    double Rr_GB = r_norm.CProb(G, B) * r_norm.GProbRecip(G, B, C) + (1.0 - r_norm.CProb(G, B)) * r_norm.GProbRecip(G, B, D);
    double Rr_GG = r_norm.CProb(G, G) * r_norm.GProbRecip(G, G, C) + (1.0 - r_norm.CProb(G, G)) * r_norm.GProbRecip(G, G, D);

    // IC(Rd_GG, Rd_GB, Rd_BG, Rd_BB);
    // IC(Rr_GG, Rr_GB, Rr_BG, Rr_BB);

    // A = R_1(G,G)+R_2(G,G) -R_1(G,B)-R_2(G,B) -R_1(B,G)-R_2(B,G) + R_1(B,B)+R_2(B,B)
    double a = Rd_GG + Rr_GG - Rd_GB - Rr_GB - Rd_BG - Rr_BG + Rd_BB + Rr_BB;
    // B = R_1(G,B)+R_2(G,B) +R_1(B,G)+R_2(B,G) -2R_1(B,B) -2R_2(B,B) -2
    double b = Rd_GB + Rr_GB + Rd_BG + Rr_BG - 2.0 * Rd_BB - 2.0 * Rr_BB - 2.0;
    // C = R_1(B,B)+R_2(B,B)
    double c = Rd_BB + Rr_BB;

    // IC(a,b,c);

    const double tolerance = 1.0e-6;
    if (a > tolerance) {
      double h_star = (-b - std::sqrt(b*b - 4.0*a*c)) / (2.0*a);
      return h_star;
    } else if (a < -tolerance) {
      double h_star = (-b - std::sqrt(b*b - 4.0*a*c)) / (2.0*a);
      return h_star;
    }
    else {
      double h_star = -c / b;
      return h_star;
    }
  }

  std::array<double,2> ESSBenefitRange() const {  // range of b for which the norm is ESS against possible mutants
    double b_lower_bound = 1.0;
    double b_upper_bound = std::numeric_limits<double>::max();
    for (int id = 0; id < 16; id++) {
      if (norm.P.ID() == id) continue;
      ActionRule mut = ActionRule::MakeDeterministicRule(id);
      auto b_range = ESSBenefitRange(mut);
      if (b_range[0] > b_lower_bound) b_lower_bound = b_range[0];
      if (b_range[1] < b_upper_bound) b_upper_bound = b_range[1];
    }
    return {b_lower_bound, b_upper_bound};
  }

  std::array<double,2> ESSBenefitRange(const ActionRule& mut) const {
    double b_lower_bound = 1.0;
    double b_upper_bound = std::numeric_limits<double>::max();

    auto p = MutantCooperationProbs(mut);
    double pc_res_mut = p.first, pc_mut_res = p.second;

    if (pc_res_res > pc_res_mut) {
      // b/c > { p_c^{\rm res \to mut} - p_c^{\rm mut \to res} / { p_c^{\rm res \to res} - p_c^{\rm res \to mut} }
      b_lower_bound = (pc_res_mut - pc_mut_res) / (pc_res_res - pc_res_mut);
      b_upper_bound = std::numeric_limits<double>::max();
    }
    else if (pc_res_res < pc_res_mut) {
      // b/c < { p_c^{\rm res \to mut} - p_c^{\rm mut \to res} / { p_c^{\rm res \to res} - p_c^{\rm res \to mut} }
      b_upper_bound = (pc_res_mut - pc_mut_res) / (pc_res_res - pc_res_mut);
      b_lower_bound = 1.0;
    }
    else {
      if (pc_res_mut >= pc_mut_res) {
        // no ESS
        b_lower_bound = std::numeric_limits<double>::max();
        b_upper_bound = 1.0;
      }
      else {
        // ESS for all
        b_lower_bound = 1.0;
        b_upper_bound = std::numeric_limits<double>::max();
      }
    }
    return {b_lower_bound, b_upper_bound};
  }

  double MutantPayoff(const ActionRule& mut, double benefit) const {
    auto pcs = MutantCooperationProbs(mut);
    double pc_res_mut = pcs.first;
    double pc_mut_res = pcs.second;
    double pi_mut = pc_res_mut * benefit - pc_mut_res * 1.0;
    return pi_mut;
  }

  std::pair<double,double> MutantCooperationProbs(const ActionRule& mut) const {
    using Reputation::B, Reputation::G, Action::C, Action::D;
    double H_star = MutantEqReputation(mut);
    const ActionRule Pmut = mut.RescaleWithError(mu_e);
    double pc_res_mut = h_star * H_star * r_norm.CProb(G, G)
        + h_star * (1.0-H_star) * r_norm.CProb(G, B)
        + (1.0-h_star) * H_star * r_norm.CProb(B, G)
        + (1.0-h_star) * (1.0-H_star) * r_norm.CProb(B, B);
    double pc_mut_res = H_star * h_star * Pmut.CProb(G, G)
        + H_star * (1.0-h_star) * Pmut.CProb(G, B)
        + (1.0-H_star) * h_star * Pmut.CProb(B, G)
        + (1.0-H_star) * (1.0-h_star) * Pmut.CProb(B, B);
    return std::make_pair(pc_res_mut, pc_mut_res);
  }

  double MutantEqReputation(const ActionRule& mut) const {
    // H^{\ast} =
    //   h^{\ast} [ R_1(B,G,P') + R_2(G,B,P) ] + (1-h^{\ast}) [ R_1(B,B,P') + R_2(B,B,P)
    //   / { 2 - h^{\ast} [R_1(G,G,P') + R_2(G,G,P) - R_1(B,G,P') - R_2(G,B,P) ] - (1-h^{\ast}) [ R_1(G,B,P') + R_2(B,G,P) - R_1(B,B,P') - R_2(B,B,P)] }.
    using Reputation::B, Reputation::G, Action::C, Action::D;
    const ActionRule Pmut = mut.RescaleWithError(mu_e);

    double r1_BBPmut = Pmut.CProb(B, B) * r_norm.GProbDonor(B, B, C) + (1.0 - Pmut.CProb(B, B)) * r_norm.GProbDonor(B, B, D);
    double r1_BGPmut = Pmut.CProb(B, G) * r_norm.GProbDonor(B, G, C) + (1.0 - Pmut.CProb(B, G)) * r_norm.GProbDonor(B, G, D);
    double r1_GBPmut = Pmut.CProb(G, B) * r_norm.GProbDonor(G, B, C) + (1.0 - Pmut.CProb(G, B)) * r_norm.GProbDonor(G, B, D);
    double r1_GGPmut = Pmut.CProb(G, G) * r_norm.GProbDonor(G, G, C) + (1.0 - Pmut.CProb(G, G)) * r_norm.GProbDonor(G, G, D);

    double r2_BBPres = r_norm.CProb(B, B) * r_norm.GProbRecip(B, B, C) + (1.0 - r_norm.CProb(B, B)) * r_norm.GProbRecip(B, B, D);
    double r2_BGPres = r_norm.CProb(B, G) * r_norm.GProbRecip(B, G, C) + (1.0 - r_norm.CProb(B, G)) * r_norm.GProbRecip(B, G, D);
    double r2_GBPres = r_norm.CProb(G, B) * r_norm.GProbRecip(G, B, C) + (1.0 - r_norm.CProb(G, B)) * r_norm.GProbRecip(G, B, D);
    double r2_GGPres = r_norm.CProb(G, G) * r_norm.GProbRecip(G, G, C) + (1.0 - r_norm.CProb(G, G)) * r_norm.GProbRecip(G, G, D);

    double num = h_star * (r1_BGPmut + r2_GBPres) + (1.0 - h_star) * (r1_BBPmut + r2_BBPres);
    double den = 2.0 - h_star * (r1_GGPmut + r2_GGPres - r1_BGPmut - r2_GBPres) - (1.0 - h_star) * (r1_GBPmut + r2_BGPres - r1_BBPmut - r2_BBPres);
    return num / den;
  }
};

#endif