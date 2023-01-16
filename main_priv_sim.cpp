#include <iostream>
#include <cassert>
#include <icecream.hpp>
#include <vector>
#include <random>
#include "Norm.hpp"
#include "PrivRepGame.hpp"


constexpr Reputation B = Reputation::B, G = Reputation::G;
constexpr Action C = Action::C, D = Action::D;


void ComprehensiveSearchWithoutR2() {
  EvolPrivRepGame::SimulationParameters params;
  params.n_init = 1e5;
  params.n_steps = 1e5;
  EvolPrivRepGameAllCAllD evol(30, params, 5.0, 1.0);

  std::vector< std::pair<int,double> > results;

  for (int j = 0; j < 256; j++) {
    std::cerr << "j = " << j << std::endl;
    AssessmentRule R1 = AssessmentRule::MakeDeterministicRule(j);
    AssessmentRule R2 = AssessmentRule::KeepRecipient();

    for (int i = 0; i < 16; i++) {
      ActionRule P = ActionRule::MakeDeterministicRule(i);
      Norm norm(R1, R2, P);
      if (norm.ID() < norm.SwapGB().ID()) {
        continue;
      }

      auto pc_rho_eq = evol.EquilibriumCoopLevelAllCAllD(norm);
      double self_cooperation_level = std::get<0>(pc_rho_eq);
      auto eq = std::get<2>(pc_rho_eq);
      double c = self_cooperation_level * eq[0] + 1.0 * eq[1];
      results.emplace_back(norm.ID(), c);
    }
  }

  // sort results by cooperation level
  std::sort(results.begin(), results.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
  // print top 10 from results
  for (int i = 0; i < 100; i++) {
    Norm norm = Norm::ConstructFromID(results[i].first);
    std::cout << norm.Inspect() << " " << results[i].second << std::endl;
  }
}

void ComprehensiveSearchWithR2() {
  EvolPrivRepGame::SimulationParameters params;
  params.n_init = 1e5;
  params.n_steps = 1e5;
  EvolPrivRepGameAllCAllD evol(30, params, 5.0, 1.0);

  std::vector< std::pair<int,double> > results;

  for (int j = 0; j < 65536; j++) {
    std::cerr << "j = " << j << std::endl;
    AssessmentRule R1 = AssessmentRule::MakeDeterministicRule(j / 256);
    AssessmentRule R2 = AssessmentRule::MakeDeterministicRule(j % 256);

    for (int i = 0; i < 16; i++) {
      ActionRule P = ActionRule::MakeDeterministicRule(i);
      Norm norm(R1, R2, P);
      if (norm.ID() < norm.SwapGB().ID()) {
        continue;
      }

      auto pc_rho_eq = evol.EquilibriumCoopLevelAllCAllD(norm);
      double self_cooperation_level = std::get<0>(pc_rho_eq);
      auto eq = std::get<2>(pc_rho_eq);
      double c = self_cooperation_level * eq[0] + 1.0 * eq[1];
      results.emplace_back(norm.ID(), c);
    }
  }

  // sort results by cooperation level
  std::sort(results.begin(), results.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
  // print top 10 from results
  for (int i = 0; i < 100; i++) {
    Norm norm = Norm::ConstructFromID(results[i].first);
    std::cout << norm.Inspect() << " " << results[i].second << std::endl;
  }
}


int main() {
  ComprehensiveSearchWithoutR2();

  return 0;
}