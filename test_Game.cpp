#include <iostream>
#include <cassert>
#include <icecream.hpp>
#include "Game.hpp"


template <typename T>
bool IsAllClose(T a, T b, double epsilon = 0.0001) {
  for (size_t i = 0; i < a.size(); i++) {
    if (std::abs(a[i] - b[i]) > epsilon) {
      return false;
    }
  }
  return true;
}

void test_Game() {
  double mu_e = 0.001, mu_a_donor = 0.001, mu_a_recip = 0.001;
  std::vector<Norm> norms = {Norm::L1(), Norm::L2(), Norm::L3(), Norm::L4(), Norm::L5(), Norm::L6(), Norm::L7(), Norm::L8()};
  for (auto n: norms) {
    Game g(mu_e, mu_a_donor, mu_a_recip, n);
    std::cerr << g.r_norm.Inspect();
    IC(g.h_star, g.pc_res_res);
    assert( g.h_star > 0.99 );
    assert( g.pc_res_res > 0.99 );
  }
}


int main() {
  test_Game();

  return 0;
}