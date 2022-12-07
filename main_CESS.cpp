#include <iostream>
#include <vector>
#include <set>
#include <map>
#include "Norm.hpp"
#include "Game.hpp"


using Reputation::G, Reputation::B, Action::C, Action::D;

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

std::array<double,2> AnalyticBenefitRange(const Norm& norm) {

  if (!norm.P.IsDeterministic()) {
    throw std::runtime_error("action rule is not deterministic");
  }

  auto R1 = [&norm](Reputation X, Reputation Y, Action A) -> double {
    return norm.Rd.GProb(X, Y, A);
  };
  auto R2 = [&norm](Reputation X, Reputation Y, Action A) -> double {
    return norm.Rr.GProb(X, Y, A);
  };
  auto P = [&norm](Reputation X, Reputation Y) -> double {
    return norm.P.CProb(X, Y);
  };
  std::cerr << norm.Inspect();

  // R_1(G,G,C) = 1 && R_2(G,G,C) = 1 &&
  // R_1(G,B) + R_2(G,B) + R_1(B,G) + R_2(B,G) > 2  => h* = 1
  Action bg = P(B, G) == 1.0 ? C : D;
  double r = R1(G,B,D) + R2(G,B,D)
      + R1(B,G,bg) + R2(B,G,bg);
  if (R1(G,G,C) == 1.0 &&
      R2(G,G,C) == 1.0 &&
      r > 2.0) {
    // h_star == 1 should be true
  }
  else {
    std::cerr << "R1(G,G,C) = " << R1(G,G,C) << std::endl;
    std::cerr << "R2(G,G,C) = " << R2(G,G,C) << std::endl;
    std::cerr << "r = " << r << std::endl;
    throw std::runtime_error("h_star is not 1");
  }

  // P(G,G) = 1
  if (P(G,G) == 1.0) {
    // game.pc_res_res == 1 should be true
  }
  else {
    throw std::runtime_error("P(G,G) is not 1");
  }

  // P(G,B) = 0
  if (P(G,B) != 0.0) {
    throw std::runtime_error("P(G,B) == 0 is necessary");
  }

  std::array<double,2> b_range = {0.0, std::numeric_limits<double>::max()};

  // if P(B,G) = 1
  if (P(B,G) == 1.0) {

    // R_1(B,G,C) > R_1(B,G,D) &&
    // b/c > (R_1(B,G,C) + R_2(G,B) ) / (R_1(B,G,C) - R_1(B,G,D))
    if (R1(B,G,C) > R1(B,G,D) ) {
      double b_lower = (R1(B, G, C) + R2(G, B, D)) / (R1(B, G, C) - R1(B, G, D));
      std::cerr << "BG: b_lower = " << b_lower << std::endl;
      if (b_lower > b_range[0]) {
          b_range[0] = b_lower;
      }
    }
    else {
      b_range[0] = std::numeric_limits<double>::max();
      b_range[1] = 0.0;
      std::cerr << "BG: R_1(B,G,C) > R_1(B,G,D) is necessary" << std::endl;
      return b_range;
    }

    // R_1(G,G,D) < 1   &&
    // b/c > ( R_1(B,G,C) + R_2(G,B) ) / ( R_1(G,G,C) - R_1(G,G,D) )
    if (R1(G,G,D) < 1.0) {
      double b_lower = (R1(B, G, C) + R2(G, B, D)) / (R1(G, G, C) - R1(G, G, D));
      std::cerr << "GG: b_lower = " << b_lower << std::endl;
      if (b_lower > b_range[0]) {
          b_range[0] = b_lower;
      }
    }
    else {
      b_range[0] = std::numeric_limits<double>::max();
      b_range[1] = 0.0;
      std::cerr << "GG: R_1(G,G,D) < 1 is necessary" << std::endl;
      return b_range;
    }

    // R_1(G,B,C) <= R_1(G,B,D) ||
    // b/c < (R_1(B,G,C) + R_2(G,B) ) / ( R_1(G,B,C) - R_1(G,B,D) )
    if (R1(G,B,C) <= R1(G,B,D)) {
      double b_upper = std::numeric_limits<double>::infinity();
      std::cerr << "GB: b_upper = " << b_upper << std::endl;
    }
    else {
      double b_upper = (R1(B, G, C) + R2(G, B, D)) / (R1(G, B, C) - R1(G, B, D));
      std::cerr << "GB: b_upper = " << b_upper << std::endl;
      if (b_upper < b_range[1]) {
        b_range[1] = b_upper;
      }
    }

    // check P(B,B) is the optimal or not
    if ( P(B,B) == 1 ) {
      // R_1(B,B,C) > R_1(B,B,D)  &&
      // b/c > {R_1(B,G,C) + R_2(G,B)} / {R_1(B,B,C) - R_1(B,B,D)}
      if (R1(B,B,C) <= R1(B,B,D)) {
        b_range[0] = std::numeric_limits<double>::max();
        b_range[1] = 0.0;
        std::cerr << "BB: R_1(B,B,C) <= R_1(B,B,D) is necessary for P(B,B)=1 to be optimal" << std::endl;
        return b_range;
      }
      else {
        double b_lower = (R1(B, G, C) + R2(G, B, D)) / (R1(B, B, C) - R1(B, B, D));
        std::cerr << "BB: b_lower = " << b_lower << std::endl;
        if (b_lower > b_range[0]) {
          b_range[0] = b_lower;
        }
      }
    }
    else if ( P(B,B) == 0 ) {
      if (R1(B,B,C) <= R1(B,B,D)) { // always ESS
        std::cerr << "BB: always ESS" << std::endl;
      }
      else {
        double b_upper = (R1(B, G, C) + R2(G, B, D)) / (R1(B, B, C) - R1(B, B, D));
        std::cerr << "BB: b_upper = " << b_upper << std::endl;
        if (b_upper < b_range[1]) {
          b_range[1] = b_upper;
        }
      }
    }
  }
  else if (P(B,G) == 0) {

    // R_1(B,G,C) \leq R_1(B,G,D)  ||
    // b/c < {R_1(B,G,C)+R_2(G,B)} / {R_1(B,G,C)-R_1(B,G,D)}
    if (R1(B,G,C) <= R1(B,G,D) ) {
      std::cerr << "BG: always ESS" << std::endl;
    }
    else {
      double b_upper = (R1(B, G, C) + R2(G, B, D)) / (R1(B, G, C) - R1(B, G, D));
      std::cerr << "BG: b_upper = " << b_upper << std::endl;
      if (b_upper < b_range[1]) {
        b_range[1] = b_upper;
      }
    }

    // R_1(G,G,D) < 1 &&
    // b/c > 1 + {R_1(B,G,D)+R_2(G,B)} / {1-R_1(G,G,D)}
    if (R1(G,G,D) < 1.0) {
      double b_lower = 1.0 + (R1(B, G, D) + R2(G, B, D)) / (1.0 - R1(G, G, D));
      std::cerr << "GG: b_lower = " << b_lower << std::endl;
      if (b_lower > b_range[0]) {
        b_range[0] = b_lower;
      }
    }
    else {
      b_range[0] = std::numeric_limits<double>::max();
      b_range[1] = 0.0;
      std::cerr << "GG: R_1(G,G,D) < 1 is necessary" << std::endl;
    }

    // R_1(G,B,D) - R_1(G,B,C) \geq 0 ||
    // b/c < 1 + {R_1(B,G,D)+R_2(G,B)} / {R_1(G,B,C) - R_1(G,B,D)}
    if (R1(G,B,D) - R1(G,B,C) >= 0.0) {
      std::cerr << "GB: always ESS" << std::endl;
    }
    else {
      double b_upper = 1.0 + (R1(B, G, D) + R2(G, B, D)) / (R1(G, B, C) - R1(G, B, D));
      std::cerr << "GB: b_upper = " << b_upper << std::endl;
      if (b_upper < b_range[1]) {
        b_range[1] = b_upper;
      }
    }

    // check P(B,B) is the optimal or not
    if ( P(B,B) == 1 ) {
      // R_1(B,B,C) > R_1(B,B,D)  AND
      // b/c > 1 + {R_1(B,G,D) + R_2(G,B) } / {R_1(B,B,C) - R_1(B,B,D)}

      if (R1(B,B,C) <= R1(B,B,D)) {
        b_range[0] = std::numeric_limits<double>::max();
        b_range[1] = 0.0;
        std::cerr << "BB: R_1(B,B,C) <= R_1(B,B,D) is necessary for P(B,B)=1 to be optimal" << std::endl;
      }
      else {
        double b_lower = 1.0 + (R1(B, G, D) + R2(G, B, D)) / (R1(B, B, C) - R1(B, B, D));
        std::cerr << "BB: b_lower = " << b_lower << std::endl;
        if (b_lower > b_range[0]) {
          b_range[0] = b_lower;
        }
      }
    }
    else if ( P(B,B) == 0 ) {
      if (R1(B,B,C) <= R1(B,B,D)) { // always ESS
        std::cerr << "BB: always ESS" << std::endl;
      }
      else {
        double b_upper = 1.0 + (R1(B, G, D) + R2(G, B, D)) / (R1(B, B, C) - R1(B, B, D));
        std::cerr << "BB: b_upper = " << b_upper << std::endl;
        if (b_upper < b_range[1]) {
          b_range[1] = b_upper;
        }
      }
    }
  } else {
    throw std::runtime_error("P(B,G) == 0 or 1 is necessary");
  }

  return b_range;
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
    auto b_range = game.StableBenefitRangeAgainstMutant(mut);
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

  Norm norm = Norm::L1();
  norm.Rd.SetGProb(G, B, D, 0.5);
  norm.Rd.SetGProb(G, B, C, 0.7);
  norm.Rd.SetGProb(B, G, C, 0.7);
  norm.Rd.SetGProb(B, G, D, 0.3);
  norm.Rd.SetGProb(G, G, D, 0.25);
  norm.Rd.SetGProb(B, B, C, 0.75);
  norm.Rd.SetGProb(B, B, D, 0.5);
  /*
  Norm norm = Norm::L1();
  norm.Rd.SetGProb(B, G, D, 1.0);
  norm.Rd.SetGProb(B, B, C, 0.0);
  norm.Rd.SetGProb(B, B, D, 1.0);
  norm.P.SetCProb(B, G, 0.0);
  norm.P.SetCProb(B, B, 0.0);
   */
  auto brange1 = AnalyticBenefitRange(norm);
  auto brange2 = Game(1.0e-6, 1.0e-6, 1.0e-6, norm).ESSBenefitRange();
  IC(brange1, brange2);

  // FindMutant();

  return 0;
}