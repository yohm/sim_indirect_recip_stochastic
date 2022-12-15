#include <iostream>
#include <cassert>
#include <chrono>
#include <icecream.hpp>
#include "PrivRepGame.hpp"


template <typename T>
bool IsAllClose(T a, T b, double epsilon = 0.02) {
  for (size_t i = 0; i < a.size(); i++) {
    if (std::abs(a[i] - b[i]) > epsilon) {
      return false;
    }
  }
  return true;
}

bool IsClose(double a, double b, double epsilon = 0.02) {
  return std::abs(a - b) < epsilon;
}

void test_SelfCooperationLevel(const Norm& norm, double expected_c_level, double expected_good_rep) {
  auto start = std::chrono::high_resolution_clock::now();

  PrivateRepGame priv_game( {{norm, 50}}, 123456789ull);
  priv_game.Update(1e4, 0.9, 0.05, false);
  priv_game.ResetCounts();
  priv_game.Update(1e4, 0.9, 0.05, true);
  IC( priv_game.NormCooperationLevels(), priv_game.NormAverageReputation() );
  assert( IsClose(priv_game.SystemWideCooperationLevel(), expected_c_level, 0.02) );
  assert( IsAllClose(priv_game.NormCooperationLevels()[0], {expected_c_level}, 0.02) );
  assert( IsAllClose(priv_game.NormAverageReputation()[0], {expected_good_rep}, 0.02) );

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cerr << "Elapsed time: " << elapsed.count() << " s\n";
}

void test_RandomNorm() {
  // random norm
  test_SelfCooperationLevel(Norm::Random(), 0.5, 0.5);
  std::cerr << "test_RandomNorm passed" << std::endl;
}


void test_LeadingEight() {

  // measure time
  test_SelfCooperationLevel(Norm::L1(), 0.90, 0.90);
  std::cerr << "test L1 passed" << std::endl;

  test_SelfCooperationLevel(Norm::L2(), 0.66, 0.65);
  std::cerr << "test L2 passed" << std::endl;

  test_SelfCooperationLevel(Norm::L3(), 0.90, 0.90);
  std::cerr << "test L3 passed" << std::endl;

  test_SelfCooperationLevel(Norm::L4(), 0.90, 0.90);
  std::cerr << "test L4 passed" << std::endl;

  test_SelfCooperationLevel(Norm::L5(), 0.70, 0.70);
  std::cerr << "test L5 passed" << std::endl;

  test_SelfCooperationLevel(Norm::L6(), 0.50, 0.50);
  std::cerr << "test L6 passed" << std::endl;

  test_SelfCooperationLevel(Norm::L7(), 0.88, 0.88);
  std::cerr << "test L7 passed" << std::endl;

  test_SelfCooperationLevel(Norm::L8(), 0.0, 0.0);
  std::cerr << "test L8 passed" << std::endl;
}

void test_SelectionMutationEquilibrium() {
  auto start = std::chrono::high_resolution_clock::now();

  Norm norm = Norm::L1();
  EvolPrivRepGame::SimulationParameters params;
  params.n_init = 1e5;
  params.n_steps = 1e5;

  EvolPrivRepGame evol(50, {norm, Norm::AllC(), Norm::AllD()}, params);
  auto rhos = evol.FixationProbabilities(5.0, 1.0);
  IC(rhos);
  assert( IsClose(rhos[0][1], 0.097, 0.02) );
  assert( IsClose(rhos[0][2], 0.000, 0.02) );
  assert( IsClose(rhos[1][0], 0.012, 0.02) );
  assert( IsClose(rhos[2][0], 0.043, 0.02) );
  auto eq = evol.EquilibriumPopulationLowMut(rhos);
  IC(eq);
  assert(IsAllClose(eq, {0.30, 0.04, 0.66}, 0.02) );

  std::cerr << "test_SelectionMutationEquilibrium passed" << std::endl;

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cerr << "Elapsed time: " << elapsed.count() << " s\n";
}


int main() {
  // test_RandomNorm();
  test_LeadingEight();
  test_SelectionMutationEquilibrium();
  return 0;
}