#include <iostream>
#include <vector>
#include <set>
#include <map>
#include "Norm.hpp"
#include "Game.hpp"


std::pair<bool,std::array<double,2>> CheckCESS(const Norm& norm, double mu_a_recip = 1.0e-3) {
  const double mu_e = 1.0e-3, mu_a_donor = 1.0e-3;
  Game game(mu_e, mu_a_donor, mu_a_recip, norm);
  auto brange = game.ESSBenefitRange();
  bool isCESS = game.pc_res_res > 0.98 && brange[0] < 1.05 && brange[1] > 100;
  // IC(brange);
  return {isCESS, brange};
}

void FindLeadingEight() {
  size_t num = 0, num_passed = 0;

  for (int j = 0; j < 256; j++) {
    AssessmentRule R1 = AssessmentRule::MakeDeterministicRule(j);
    AssessmentRule R2 = AssessmentRule::KeepRecipient();

    for (int i = 0; i < 16; i++) {
      ActionRule P = ActionRule::MakeDeterministicRule(i);
      Norm norm(R1, R2, P);
      if ( norm.ID() < norm.SwapGB().ID() ) {
        continue;
      }

      auto [isCESS, brange] = CheckCESS(norm, 0.0);
      if (isCESS) {
        if (norm.CProb(Reputation::G, Reputation::G) != 1.0) {
          norm = norm.SwapGB();
        }
        std::cerr << norm.Inspect();
        std::cerr << "brange: "  << brange[0] << " " << brange[1] << std::endl;
        num_passed++;
      }
      num++;
    }
  }
  IC(num, num_passed);
}

void EnumerateAllCESS() {
  size_t num = 0, num_passed = 0;
  std::map<std::string, std::set<int>> cess_norms;

  for (int j = 0; j < 256; j++) {
    AssessmentRule R1 = AssessmentRule::MakeDeterministicRule(j);
    for (int k = 0; k < 256; k++) {
      AssessmentRule R2 = AssessmentRule::MakeDeterministicRule(k);

      for (int i = 0; i < 16; i++) {
        ActionRule P = ActionRule::MakeDeterministicRule(i);
        Norm norm(R1, R2, P);
        if ( norm.ID() < norm.SwapGB().ID() ) {
          continue;
        }

        auto [isCESS, brange] = CheckCESS(norm);
        if (isCESS) {
          if (norm.CProb(Reputation::G, Reputation::G) != 1.0) {
            norm = norm.SwapGB();
          }
          //std::cerr << norm.Inspect();
          cess_norms[norm.SimilarNorm()].insert(norm.Rr.ID());
          std::cerr << "brange: "  << brange[0] << " " << brange[1] << std::endl;
          num_passed++;
        }
        num++;
      }
    }
  }

  IC(num, num_passed);
  for (auto& kv : cess_norms) {
    std::cerr << kv.first << " " << kv.second.size() << std::endl;
    for (auto& v : kv.second) {
      std::cerr << std::bitset<8>(v) << std::endl;
    }
    std::cerr << std::endl;
  }
}

void EnumerateR2() {

  std::array<Norm,8> leading_eight = {Norm::L1(), Norm::L2(), Norm::L3(), Norm::L4(),
                                      Norm::L5(), Norm::L6(), Norm::L7(), Norm::L8()};
  std::set<int> R2_ids = {
      0b10001000, 0b10001001, 0b10001010, 0b10001011, 0b10001100, 0b10001101, 0b10001110, 0b10001111,
      0b10101000, 0b10101001, 0b10101010, 0b10101011, 0b10101100, 0b10101101, 0b10101110, 0b10101111,
      0b11001000, 0b11001001, 0b11001010, 0b11001011, 0b11001100, 0b11001101, 0b11001110, 0b11001111,
      0b11101000, 0b11101001, 0b11101010, 0b11101011, 0b11101100, 0b11101101, 0b11101110, 0b11101111
  };

  size_t num = 0, num_passed = 0;
  for (Norm norm: leading_eight) {
    // overwrite R2
    std::set<int> cess_ids;
    for (int k = 0; k < 256; k++) {
      AssessmentRule R2 = AssessmentRule::MakeDeterministicRule(k);
      norm.Rr = R2;

      auto [isCESS, brange] = CheckCESS(norm);
      if (isCESS) {
        cess_ids.insert(norm.Rr.ID());
        num_passed++;
      }
      num++;
    }
    if (cess_ids == R2_ids) {
      std::cout << "Identical" << std::endl;
    }
    else {
      throw std::runtime_error("something wrong");
    }
  }

  IC(num, num_passed, R2_ids.size());
}

bool CloseEnough(double a, double b, double tol = 1.0e-2) {
  return std::abs(a - b) < tol;
}

void CheckAnalyticFormula() {
  // using Reputation
  using Reputation::G, Reputation::B, Action::C, Action::D;

  Norm norm = Norm::L1();
  norm.Rd.SetGProb(G, B, D, 0.5);
  norm.Rd.SetGProb(G, B, C, 0.7);
  norm.Rd.SetGProb(B, G, D, 0.1);
  norm.Rd.SetGProb(G, G, D, 0.25);

  Game game(1.0e-3, 1.0e-3, 1.0e-3, norm);

  const AssessmentRule R1 = norm.Rd;
  const AssessmentRule R2 = norm.Rr;
  const ActionRule P = norm.P;
  std::cerr << norm.Inspect();

  // R_1(G,G,C) = 1 && R_2(G,G,C) = 1 &&
  // R_1(G,B,D) + R_2(G,B,D) + R_1(B,G,C) + R_2(B,G,C) > 2  => h* = 1
  double r = R1.GProb(G,B,D) + R2.GProb(G,B,D)
      + R1.GProb(B,G,C) + R2.GProb(B,G,C);
  if (R1.GProb(G,G,C) == 1.0 &&
      R2.GProb(G,G,C) == 1.0 &&
      r > 2.0) {
    if (!CloseEnough(game.h_star, 1.0)) {
      std::cerr << norm.Inspect();
      std::cerr << "h* = " << game.h_star << std::endl;
      assert(false);
    }
  }
  else {
    std::cerr << "r = " << r << std::endl;
    std::runtime_error("r is not greater than 2");
  }

  // P(G,G) = 1
  if (P.CProb(G,G) == 1.0) {
    if (!CloseEnough(game.pc_res_res, 1.0)) {
      std::cerr << "pc_res_res = " << game.pc_res_res << std::endl;
      assert(false);
    }
  }

  // P(G,B) = 0
  if (norm.CProb(G,B) != 0.0) {
    std::runtime_error("P(G,B) == 0 is necessary");
  }

  // P(B,G) = 1
  if (norm.CProb(B,G) == 1.0) {

    // R_1(B,G,C) > R_1(B,G,D) &&
    // b/c > (R_1(B,G,D) + R_2(G,B) ) / (R_1(B,G,C) - R_1(B,G,D))
    if (R1.GProb(B,G,C) > R1.GProb(B,G,D) ) {
      double b_lower = (R1.GProb(B, G, D) + R2.GProb(G, B, D)) / (R1.GProb(B, G, C) - R1.GProb(B, G, D));
      IC(b_lower);
    }
    else {
      std::runtime_error("R_1(B,G,C) > R_1(B,G,D) is necessary");
    }

    // R_1(G,G,D) < 1   &&
    // b/c > ( R_1(B,G,C) + R_2(G,B) ) / ( R_1(G,G,C) - R_1(G,G,D) )
    if (R1.GProb(G,G,D) < 1.0) {
      double b_lower = (R1.GProb(B, G, C) + R2.GProb(G, B, D)) / (R1.GProb(G, G, C) - R1.GProb(G, G, D));
      IC(b_lower);
    }
    else {
      std::runtime_error("R_1(G,G,D) < 1 is necessary");
    }

    // R_1(G,B,C) <= R_1(G,B,D) ||
    // b/c < (R_1(B,G,C) + R_2(G,B) ) / ( R_1(G,B,C) - R_1(G,B,D) )
    if (R1.GProb(G,B,C) <= R1.GProb(G,B,D)) {
      double b_upper = std::numeric_limits<double>::infinity();
      IC(b_upper);
    }
    else {
      double b_upper = (R1.GProb(B, G, C) + R2.GProb(G, B, D)) / (R1.GProb(G, B, C) - R1.GProb(G, B, D));
      IC(b_upper);
    }
  }


  auto [isCESS, brange] = CheckCESS(norm);
  IC(isCESS, brange);
}

void FindMutant() {
  Norm norm = Norm::L3();
  norm.Rr = AssessmentRule::MakeDeterministicRule(0b10001000);
  norm.Rr.SetGProb(Reputation::G, Reputation::B, Action::D, 1.0);

  const double mu_e = 1.0e-2, mu_a_donor = 1.0e-2, mu_a_recip = 1.0e-2;
  Game game(mu_e, mu_a_donor, mu_a_recip, norm);
  auto brange = game.ESSBenefitRange();
  IC(brange);

  for (int id = 0; id < 16; id++) {
    if (norm.P.ID() == id) continue;
    ActionRule mut = ActionRule::MakeDeterministicRule(id);
    auto b_range = game.ESSBenefitRange(mut);
    IC(id, b_range);
  }

  ActionRule alld = ActionRule::ALLD();
  double h_mut = game.MutantEqReputation(alld);
  auto probs = game.MutantCooperationProbs(alld);
  IC(h_mut, probs, game.pc_res_res);

}

int main() {
  // FindLeadingEight();
  // EnumerateAllCESS();
  // EnumerateR2();

  CheckAnalyticFormula();

  // FindMutant();

  return 0;
}