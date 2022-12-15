#include <iostream>
#include <cassert>
#include <icecream.hpp>
#include "PrivRepGame.hpp"


template <typename T>
bool IsAllClose(T a, T b, double epsilon = 1.0e-6) {
  for (size_t i = 0; i < a.size(); i++) {
    if (std::abs(a[i] - b[i]) > epsilon) {
      return false;
    }
  }
  return true;
}

bool IsClose(double a, double b, double epsilon = 1.0e-6) {
  return std::abs(a - b) < epsilon;
}

using Reputation::G, Reputation::B, Action::C, Action::D;

void test_SelfCooperationLevel(const Norm& norm, double expected_c_level, double expected_good_rep) {
  PrivateRepGame priv_game( {{norm, 50}}, 123456789ull);
  priv_game.Update(1e4, 0.9, 0.05, false);
  priv_game.ResetCounts();
  priv_game.Update(1e4, 0.9, 0.05, true);
  IC( priv_game.NormCooperationLevels(), priv_game.NormAverageReputation() );
  assert( IsClose(priv_game.SystemWideCooperationLevel(), expected_c_level, 0.02) );
  assert( IsAllClose(priv_game.NormCooperationLevels()[0], {expected_c_level}, 0.02) );
  assert( IsAllClose(priv_game.NormAverageReputation()[0], {expected_good_rep}, 0.02) );
  // priv_game.PrintImage(std::cerr);
}

void test_RandomNorm() {
  // random norm
  test_SelfCooperationLevel(Norm::Random(), 0.5, 0.5);
  std::cerr << "test_RandomNorm passed" << std::endl;
}


void test_LeadingEight() {
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


int main() {
  test_RandomNorm();
  test_LeadingEight();

  return 0;
}