#include <iostream>
#include <cassert>
#include <icecream.hpp>
#include <vector>
#include <random>
#include "Norm.hpp"
#include "PrivRepGame.hpp"


constexpr Reputation B = Reputation::B, G = Reputation::G;
constexpr Action C = Action::C, D = Action::D;


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

  Norm norm = Norm::L3();
  // norm.Rd.SetGProb(G, B, D, 0.5);
  // norm.Rd.SetGProb(G, B, C, 0.5);
  // norm.Rr = AssessmentRule::ImageScoring();
  // norm.Rr = AssessmentRule::KeepRecipient();

  PrivateRepGame priv_game( {{norm, 50}}, 123456789ull);
  priv_game.Update(1e6, 0.9, 0.05, false);
  priv_game.ResetCounts();
  priv_game.Update(1e6, 0.9, 0.05, false);
  IC( priv_game.NormCooperationLevels() );

  EvolutionaryPrivateRepGame::SimulationParameters params;

  EvolutionaryPrivateRepGame evol(50, {norm, Norm::AllC(), Norm::AllD()}, params);
  auto rhos = evol.FixationProbabilities(5.0, 1.0);
  IC(rhos);
  auto eq = evol.EquilibriumPopulationLowMut(rhos);
  IC(eq);

  return 0;
}