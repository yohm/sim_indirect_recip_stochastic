#include <iostream>
#include <cmath>
#include <vector>
#include <set>
#include <map>
#include <random>
#include <bitset>
#include "Norm.hpp"
#include "PublicRepGame.hpp"


constexpr Reputation B = Reputation::B, G = Reputation::G;
constexpr Action C = Action::C, D = Action::D;

std::tuple<bool,std::array<double,2>,double> CheckCESS(const Norm& norm, double mu_a_recip = 1.0e-3) {
  const double mu_e = 1.0e-3, mu_a_donor = 1.0e-3;
  PublicRepGame game(mu_e, mu_a_donor, mu_a_recip, norm);
  auto brange = game.ESSBenefitRange();
  bool isCESS = game.pc_res_res > 0.98 && brange[0] < 10 && brange[0] + 0.001 < brange[1];
  // IC(brange);
  return {isCESS, brange, game.h_star};
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

      auto [isCESS, brange, h_star] = CheckCESS(norm, 0.0);
      if (isCESS) {
        if (h_star < 0.5) {
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

std::string IdentifyType(const Norm& norm) {
  auto R1 = norm.Rd;
  auto R2 = norm.Rr;
  auto P = norm.P;
  // P : [100*]
  if ( P.CProb(G, G) == 1.0 && P.CProb(G, B) == 0.0 && P.CProb(B, G) == 0.0 ) {
    if ( R2.GProb(G,G,C) == 1.0 && R2.GProb(G,B,D) == 1.0 && R2.GProb(B,G,D) == 1.0 ) {
      if ( R1.GProb(G,B,D) == 1.0 && R1.GProb(B,G,D) == 0.0) {
        return "P100_R2_111_R1_10";
      }
      else if ( R1.GProb(G,B,D) == 0.0 && R1.GProb(B,G,D) == 1.0) {
        return "P100_R2_111_R1_01";
      }
    }
  }
  // P : [101*]
  if ( P.CProb(G, G) == 1.0 && P.CProb(G, B) == 0.0 && P.CProb(B, G) == 1.0 ) {
    if (R2.GProb(G, G, C) == 1.0 && R2.GProb(G, B, D) == 1.0 && R2.GProb(B, G, C) == 1.0) {
      return "P101_R2_111";
    }
  }
  return "other";
}

void EnumerateAllCESS() {
  size_t num = 0, num_passed = 0;
  std::map< std::pair<std::string,int>, std::vector<int>> cess_norms;

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

        auto [isCESS, brange, h_star] = CheckCESS(norm);
        if (isCESS) {
          if (h_star < 0.5) {
            norm = norm.SwapGB();
          }
          //std::cerr << norm.Inspect();
          std::string similar_norm_name = norm.SimilarNorm();
          if (similar_norm_name.empty()) {
            similar_norm_name = IdentifyType(norm);
          }
          if (brange[1] < 100) {
            throw std::runtime_error("brange[1] < 100");
          }
          cess_norms[std::make_pair(similar_norm_name,std::round(brange[0]))].push_back(norm.ID());
          // std::cerr << "brange: "  << brange[0] << " " << brange[1] << std::endl;
          num_passed++;
        }
        num++;
      }
    }
  }

  IC(num, num_passed);
  for (auto& kv : cess_norms) {
    std::cerr << kv.first.first << " " << kv.first.second << " " << kv.second.size() << std::endl;
    if (kv.first.first == "other") {
      for (auto& v : kv.second) {
        std::cerr << Norm::ConstructFromID(v).Inspect();
        break;
        // std::cerr << std::bitset<8>(v) << std::endl;
      }
    }
  }
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

Norm MakeNormFromTable(std::array<Action,4> actions, std::array<double,8> r1, std::array<double,8> r2 = {1,1,0,0,1,1,0,0}) {
  ActionRule ar({
                    {{G,G}, actions[0]==C?1:0},
                    {{G,B}, actions[1]==C?1:0},
                    {{B,G}, actions[2]==C?1:0},
                    {{B,B}, actions[3]==C?1:0},
  });

  AssessmentRule Rd({
                        {{G,G,C}, r1[0]}, {{G,G,D}, r1[1]},
                        {{G,B,C}, r1[2]}, {{G,B,D}, r1[3]},
                        {{B,G,C}, r1[4]}, {{B,G,D}, r1[5]},
                        {{B,B,C}, r1[6]}, {{B,B,D}, r1[7]},
  });
  AssessmentRule Rr({
                        {{G,G,C}, r2[0]}, {{G,G,D}, r2[1]},
                        {{G,B,C}, r2[2]}, {{G,B,D}, r2[3]},
                        {{B,G,C}, r2[4]}, {{B,G,D}, r2[5]},
                        {{B,B,C}, r2[6]}, {{B,B,D}, r2[7]},
                    });
  return Norm(Rd, Rr, ar);
}

void StochasticVariantLeadingEight() {
  {
    double p3 = 0.3;
    double p1 = 1.0 - p3;
    double p2 = 1.5 - p3;
    Norm norm = MakeNormFromTable({C, D, C, C}, {1, p1, 0, p2, p3, 0, 1, 0}, {1, 1, 0, 0, 1, 1, 0, 0});

    auto brange1 = AnalyticBenefitRange(norm);
    auto brange2 = PublicRepGame(1.0e-6, 1.0e-6, 1.0e-6, norm).ESSBenefitRange();
    IC(brange1, brange2);
  }

  {
    double p3 = 0.2;
    double p1 = 0.0;
    double p2 = 1.0;
    Norm norm = MakeNormFromTable({C, D, D, D}, {1, p1, 0, p2, 0, p3, 0, 0}, {1, 1, 0, 0, 1, 1, 0, 0});
    auto brange1 = AnalyticBenefitRange(norm);
    auto brange2 = PublicRepGame(1.0e-6, 1.0e-6, 1.0e-6, norm).ESSBenefitRange();
    IC(brange1, brange2);
  }
}

bool CompareAnalyticNumericalBranges(const Norm& norm) {
  std::cerr << norm.Inspect();
  auto brange1 = AnalyticBenefitRange(norm);
  auto brange2 = PublicRepGame(1.0e-6, 1.0e-6, 1.0e-6, norm).ESSBenefitRange();
  IC(brange1, brange2);
  double th = 3.0e-2;
  if (brange1[0] > brange1[1]) {
    return brange2[0] > brange2[1];
  }
  return ( std::abs(brange1[0] - brange2[0]) < brange1[0]*th && std::abs(brange1[1] - brange2[1]) < brange1[1]*th );
}

void RandomCheckAnalyticNorms() {
  size_t num_norms = 100000;
  std::mt19937_64 rng(123456789);
  std::uniform_real_distribution<double> dist(0.0, 1.0);
  auto r01 = [&rng,&dist]() -> double {
    return dist(rng);
  };

  std::vector<Norm> unpassed;
  for (size_t i = 0; i < num_norms; i++) {
    std::cerr << std::endl << "i: " << i << std::endl;
    Action P_bg = (rng() % 2 == 0) ? C : D;
    Action P_bb = (rng() % 2 == 0) ? C : D;
    std::array<Action,4> actions = {C,D,P_bg,P_bb};

    std::array<double,8> r1 = {
        1.0,   r01(), // GG
        r01(), r01(), // GB
        r01(), r01(), // BG
        r01(), r01()  // BB
    };
    std::array<double,8> r2 = {
        1.0,   r01(), // GG
        r01(), r01(), // GB
        r01(), r01(), // BG
        r01(), r01()  // BB
    };

    // R_1(G,B,D) + R_2(G,B,D) + R_1(B,G,C) + R_2(B,G,C) > 2 is necessary
    double recov = (P_bg==C) ? (r1[3] + r2[3] + r1[4] + r2[4]) : (r1[3] + r2[3] + r1[5] + r2[5]);
    if (recov <= 2.0) { std::cerr << "does not satisfy recov > 2. continue." << std::endl; continue; }

    Norm norm = MakeNormFromTable(actions, r1, r2);
    bool b = CompareAnalyticNumericalBranges(norm);
    if (!b) {
      unpassed.push_back(norm);
      // throw std::runtime_error("Failed");
    }

  }

  std::cerr << "Number of unpassed: " << unpassed.size() << std::endl;
  for (auto& norm : unpassed) {
    bool b = CompareAnalyticNumericalBranges(norm);
    std::cerr << "-----------------------------------------------------------\n";
    std::cerr << PublicRepGame(1.0e-6, 1.0e-6, 1.0e-6, norm).Inspect();
    std::cerr << "-----------------------------------------------------------\n\n";
  }

}

int main() {
  FindLeadingEight();
  // EnumerateAllCESS();
  // StochasticVariantLeadingEight();
  // RandomCheckAnalyticNorms();

  return 0;
}
