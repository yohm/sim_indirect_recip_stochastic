#include <iostream>
#include <cassert>
#include <icecream.hpp>
#include "Norm.hpp"


template <typename T>
bool IsAllClose(T a, T b, double epsilon = 0.0001) {
  for (size_t i = 0; i < a.size(); i++) {
    if (std::abs(a[i] - b[i]) > epsilon) {
      return false;
    }
  }
  return true;
}

void test_ActionRule() {
  ActionRule P({0.1, 0.2, 0.5, 0.9});
  std::cout << P.Inspect();

  assert( P.CProb(Reputation::B, Reputation::B) == 0.1 );
  assert( P.CProb(Reputation::G, Reputation::B) == 0.5 );
  assert(P.IsSecondOrder() == false);

  P.SetCProb(Reputation::B, Reputation::G, 0.3);
  assert( P.CProb(Reputation::B, Reputation::G) == 0.3 );

  ActionRule P2 = P.SwapGB();
  assert( P2.CProb(Reputation::G, Reputation::G) == 0.1 );
  assert( P2.CProb(Reputation::B, Reputation::G) == 0.5 );

  ActionRule P3({0.0, 1.0, 0.0, 1.0});
  std::cout << P3.Inspect();
  assert( P3.IsDeterministic() == true );
  assert( P3.ID() == 10 );
  assert( P3.IsSecondOrder() == true);

  auto disc = ActionRule::DISC();
  assert( disc.IsDeterministic() == true );
  assert( disc.ID() == 10 );
  auto allc = ActionRule::ALLC();
  assert( allc.IsDeterministic() == true );
  assert( allc.ID() == 15 );
  auto alld = ActionRule::ALLD();
  assert( alld.IsDeterministic() == true );
  assert( alld.ID() == 0 );

  auto P4 = ActionRule::MakeDeterministicRule(7);
  assert( P4.ID() == 7 );

  std::cout << "test_ActionRule passed!" << std::endl;
}

void test_AssessmentRule() {
  AssessmentRule R({0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8});
  std::cout << R.Inspect();

  assert( R.GProb(Reputation::B, Reputation::B, Action::D) == 0.1 );
  assert( R.GProb(Reputation::B, Reputation::B, Action::C) == 0.2 );
  assert( R.GProb(Reputation::G, Reputation::B, Action::C) == 0.6 );
  assert( R.IsSecondOrder() == false);

  R.SetGProb(Reputation::B, Reputation::G, Action::C, 0.9);
  assert( R.GProb(Reputation::B, Reputation::G, Action::C) == 0.9 );

  AssessmentRule R2 = R.SwapGB();
  assert( R2.GProb(Reputation::G, Reputation::G, Action::C) == 1.0 - R.GProb(Reputation::B, Reputation::B, Action::C) );
  assert( R2.GProb(Reputation::B, Reputation::G, Action::D) == 1.0 - R.GProb(Reputation::G, Reputation::B, Action::D) );

  AssessmentRule Q3({0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0});
  std::cout << Q3.Inspect();
  assert( Q3.IsDeterministic() == true );
  assert( Q3.ID() == 0b10101010 );
  assert( Q3.IsSecondOrder() == true);

  auto allg = AssessmentRule::AllGood();
  assert( allg.IsDeterministic() == true );
  assert( allg.ID() == 0b11111111 );
  auto allb = AssessmentRule::AllBad();
  assert( allb.IsDeterministic() == true );
  assert( allb.ID() == 0b00000000 );
  auto is = AssessmentRule::ImageScoring();
  assert( is.IsDeterministic() == true );
  assert( is.ID() == 0b10101010 );

  std::cout << "test_AssessmentRule passed!" << std::endl;
}

void test_Norm() {
  ActionRule p = ActionRule::DISC();
  AssessmentRule r1 = AssessmentRule::ImageScoring();
  AssessmentRule r2 = AssessmentRule::AllGood();

  std::cout << Norm(r1, r2, p).Inspect();

  Norm n(
      {{0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7}},
      {{0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}},
      {{0.0,0.25,0.5,0.75}}
      );
  assert( n.IsDeterministic() == false );
  assert( n.IsSecondOrder() == false );
  assert( n.IsRecipKeep() == false );
  assert( n.GProbDonor(Reputation::B, Reputation::G, Action::C) == 0.3 );
  assert( n.GProbRecip(Reputation::B, Reputation::G, Action::C) == 0.5 );
  assert( n.CProb(Reputation::B, Reputation::G) == 0.25 );
  std::cout << n.Inspect();

  // test Image Scoring
  {
    Norm is = Norm::ImageScoring();
    assert( is.IsRecipKeep() );
    assert( is.IsDeterministic() );
    assert( is.IsSecondOrder() );
    assert( is.ID() == 0b1010'10101010'11001100 );
  }

  // test leading eight
  std::array<Norm,8> leading_eight = {Norm::L1(), Norm::L2(), Norm::L3(), Norm::L4(),
                                      Norm::L5(), Norm::L6(), Norm::L7(), Norm::L8()};
  std::array<int,8> l8_ids = {768716, 760524, 703436, 702924, 695244, 694732, 702668, 694476};
  for (size_t i = 0; i < 8; i++) {
    auto l = leading_eight[i];
    std::cout << "L" << i+1 << " : " << l.Inspect();
    assert( l.IsDeterministic() == true );
    assert( l.IsRecipKeep() == true );
    if (i == 2 || i == 5) {
      assert( l.IsSecondOrder() == true );
    } else {
      assert( l.IsSecondOrder() == false );
    }
    assert( l.ID() == l8_ids[i] );
    assert( Norm::ConstructFromID(l8_ids[i]) == l );

    assert( l.CProb(Reputation::G, Reputation::G) == 1.0 );
    assert( l.CProb(Reputation::G, Reputation::B) == 0.0 );
    assert( l.CProb(Reputation::B, Reputation::G) == 1.0 );
    assert( l.GProbDonor(Reputation::G, Reputation::G, Action::C) == 1.0 );
    assert( l.GProbDonor(Reputation::G, Reputation::G, Action::D) == 0.0 ); // identification of defectors
    assert( l.GProbDonor(Reputation::B, Reputation::G, Action::D) == 0.0 ); // identification of defectors
    assert( l.GProbDonor(Reputation::G, Reputation::B, Action::D) == 1.0 ); // justified punishment
    assert( l.GProbDonor(Reputation::B, Reputation::G, Action::C) == 1.0 ); // apology
  }

  // test rescaling
  {
    Norm n = Norm::L6();
    auto rescaled = n.RescaleWithError(0.1, 0.02, 0.0);
    std::cout << "rescaled: " << rescaled.Inspect() << std::endl;
    auto expected1 = std::array<double,4>{0.0, 0.9, 0.0, 0.9};
    bool b1 = IsAllClose(rescaled.P.coop_probs, expected1);
    assert(b1);
    auto expected2 = std::array<double,8>{0.98, 0.02, 0.02, 0.98, 0.98, 0.02, 0.02, 0.98};
    bool b2 = IsAllClose(rescaled.Rd.good_probs, expected2 );
    assert(b2);
    auto expected3 = std::array<double,8>{0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0};
    bool b3 = IsAllClose(rescaled.Rr.good_probs, expected3 );
    assert(b3);
  }

  std::cout << "test_Norm passed!" << std::endl;
}


int main() {
  test_ActionRule();
  test_AssessmentRule();
  test_Norm();

  return 0;
}