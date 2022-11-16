#include <iostream>
#include <vector>
#include <set>
#include "Norm.hpp"
#include "Game.hpp"


bool IsCESS(const Norm& norm, double mu_a_recip = 1.0e-2) {
  const double mu_e = 1.0e-2, mu_a_donor = 1.0e-2;
  Game game(mu_e, mu_a_donor, mu_a_recip, norm);
  auto brange = game.ESSBenefitRange();
  // IC(brange);
  return (game.pc_res_res > 0.95 && brange[0] < 1.001 && brange[1] > 1000);
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

      if (IsCESS(norm, 0.0)) {
        if (norm.CProb(Reputation::G, Reputation::G) != 1.0) {
          norm = norm.SwapGB();
        }
        std::cerr << norm.Inspect();
        num_passed++;
      }
      num++;
    }
  }
  IC(num, num_passed);
}

void EnumerateAllCESS() {
  size_t num = 0, num_passed = 0;
  const double mu_e = 1.0e-2, mu_a_donor = 1.0e-2, mu_a_recip = 1.0e-2;

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

        if (IsCESS(norm, 0.0)) {
          if (norm.CProb(Reputation::G, Reputation::G) != 1.0) {
            norm = norm.SwapGB();
          }
          std::cerr << norm.Inspect();
          num_passed++;
        }
        num++;
      }
    }
  }

  IC(num, num_passed);
}

void EnumerateR2() {
  size_t num = 0, num_passed = 0;
  const double mu_e = 1.0e-2, mu_a_donor = 1.0e-2, mu_a_recip = 1.0e-2;

  std::array<Norm,8> leading_eight = {Norm::L1(), Norm::L2(), Norm::L3(), Norm::L4(),
                                      Norm::L5(), Norm::L6(), Norm::L7(), Norm::L8()};

  std::set<int> R2_ids = {
      0b10001000, 0b10001001, 0b10001010, 0b10001011, 0b10001100, 0b10001101, 0b10001110, 0b10001111,
      0b10101000, 0b10101001, 0b10101010, 0b10101011, 0b10101100, 0b10101101, 0b10101110, 0b10101111,
      0b11001000, 0b11001001, 0b11001010, 0b11001011, 0b11001100, 0b11001101, 0b11001110, 0b11001111,
      0b11101000, 0b11101001, 0b11101010, 0b11101011, 0b11101100, 0b11101101, 0b11101110, 0b11101111
  };
  for (Norm norm: leading_eight) {
    // overwrite R2
    std::set<int> cess_ids;
    for (int k = 0; k < 256; k++) {
      AssessmentRule R2 = AssessmentRule::MakeDeterministicRule(k);
      norm.Rr = R2;

      if (IsCESS(norm, 0.0)) {
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

int main() {
  // FindLeadingEight();
  EnumerateAllCESS();

  // EnumerateR2();

  return 0;
}