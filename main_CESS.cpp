#include <iostream>
#include "Norm.hpp"
#include "Game.hpp"

int main() {

  size_t num = 0, num_passed = 0;
  const double mu_e = 1.0e-2, mu_a_donor = 1.0e-2, mu_a_recip = 0.0;

  for (int j = 0; j < 256; j++) {
    AssessmentRule R1 = AssessmentRule::MakeDeterministicRule(j);
    for (int i = 0; i < 16; i++) {
      ActionRule P = ActionRule::MakeDeterministicRule(i);
      AssessmentRule R2 = AssessmentRule::KeepRecipient();
      Norm norm(R1, R2, P);
      if ( norm.ID() < norm.SwapGB().ID() ) {
        continue;
      }

      Game game(mu_e, mu_a_donor, mu_a_recip, norm);
      auto brange = game.ESSBenefitRange();
      // IC(brange);
      if (game.pc_res_res > 0.95 && brange[0] < 1.02 && brange[1] > 10) {
        Norm n = game.norm;
        if (n.CProb(Reputation::G, Reputation::G) != 1.0) {
          n = n.SwapGB();
        }
        std::cerr << n.Inspect();
        IC(brange, game.norm.ID(), n.ID());
        num_passed++;
      }
      num++;
    }
  }

  IC(num, num_passed);

  return 0;
}