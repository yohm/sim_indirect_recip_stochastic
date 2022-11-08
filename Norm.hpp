#ifndef NORM_HPP
#define NORM_HPP

#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <array>
#include <cstdint>


enum class Action {
  D = 0,  // defect
  C = 1   // cooperate
};

Action FlipAction(Action a) {
  if (a == Action::C) { return Action::D; }
  else { return Action::C; }
}

enum class Reputation {
  B = 0,   // bad
  G = 1    // good
};

char A2C(Action a) {
  if (a == Action::D) { return 'd'; }
  else { return 'c'; }
}
Action C2A(char c) {
  if (c == 'd') { return Action::D; }
  else if (c == 'c') { return Action::C; }
  else { throw std::runtime_error("invalid character for action"); }
}
char R2C(Reputation r) {
  if (r == Reputation::B) { return 'B'; }
  else { return 'G'; }
}
Reputation C2R(char c) {
  if (c == 'B') { return Reputation::B; }
  else if (c == 'G') { return Reputation::G; }
  else { throw std::runtime_error("invalid character for action"); }
}
std::ostream &operator<<(std::ostream &os, const Action &act) {
  os << A2C(act);
  return os;
}
std::ostream &operator<<(std::ostream &os, const Reputation &rep) {
  os << R2C(rep);
  return os;
}

class ActionRule {
  public:
  ActionRule(const std::array<double,4>& coop_probs) : coop_probs(coop_probs) {};
  // {P(C|B,B), P(C|B,G), P(C|G,B), P(C|G,G)}

  ActionRule Clone() const { return ActionRule(coop_probs); }
  double CProb(const Reputation& rep_d, const Reputation& rep_r) const {
    size_t idx = static_cast<size_t>(rep_d) * 2 + static_cast<size_t>(rep_r);
    return coop_probs[idx];
  }
  void SetCProb(const Reputation& rep_d, const Reputation& rep_r, double c_prob) {
    size_t idx = static_cast<size_t>(rep_d) * 2 + static_cast<size_t>(rep_r);
    coop_probs[idx] = c_prob;
  }
  ActionRule SwapGB() const { // returns a new action rule with G and B swapped
    std::array<double,4> new_coop_probs = {coop_probs[3], coop_probs[2], coop_probs[1], coop_probs[0]};
    return ActionRule{new_coop_probs};
  }

  std::string Inspect() const {
    std::stringstream ss;
    ss << "ActionRule:" << std::endl;
    for (size_t i = 0; i < 4; i++) {
      Reputation rep_d = static_cast<Reputation>(i / 2);
      Reputation rep_r = static_cast<Reputation>(i % 2);
      ss << "(" << rep_d << "->" << rep_r << ") : " << coop_probs[i];
      if (i % 2 == 1) { ss << std::endl; }
      else { ss << "\t"; }
    }
    return ss.str();
  }

  bool IsSecondOrder() const {
    if (coop_probs[0] == coop_probs[2] && coop_probs[1] == coop_probs[3]) { return true; }
    else { return false; }
  }

  std::array<double,4> coop_probs; // P(C|B,B), P(C|B,G), P(C|G,B), P(C|G,G)

  static ActionRule MakeDeterministicRule(int id) {
    if (id < 0 || id > 15) {
      throw std::runtime_error("AssessmentRuleDet: id must be between 0 and 15");
    }
    std::array<double,4> c_probs = {0.0};
    for (size_t i = 0; i < 8; i++) {
      if ((id >> i) % 2) { c_probs[i] = 1.0; }
      else { c_probs[i] = 0.0; }
    }
    return ActionRule{ c_probs };
  }
};


class AssessmentRule {
  public:
  AssessmentRule(const std::array<double,8>& g_probs) : good_probs(g_probs) {};
  // {P(G|B,B,D), P(G|B,B,C), P(G|B,G,D), P(G|B,G,C), P(G|G,B,D), P(G|G,B,C), P(G|G,G,D), P(G|G,G,C)}

  AssessmentRule Clone() const { return AssessmentRule(good_probs); }
  AssessmentRule SwapGB() const {
    std::array<double,8> new_good_probs = {good_probs[6], good_probs[7], good_probs[4], good_probs[5],
                                           good_probs[2], good_probs[3], good_probs[0], good_probs[1]};
    return {new_good_probs};
  }
  double GProb(const Reputation& rep_d, const Reputation& rep_r, const Action& act) const {
    size_t idx = static_cast<size_t>(rep_d) * 4 + static_cast<size_t>(rep_r) * 2 + static_cast<size_t>(act);
    return good_probs[idx];
  }
  void SetGProb(const Reputation& rep_d, const Reputation& rep_r, const Action& act, double g_prob) {
    size_t idx = static_cast<size_t>(rep_d) * 4 + static_cast<size_t>(rep_r) * 2 + static_cast<size_t>(act);
    good_probs[idx] = g_prob;
  }

  std::string Inspect() const {
    std::stringstream ss;
    ss << "AssessmentRule:" << std::endl;
    for (size_t i = 0; i < 8; i++) {
      Reputation rep_d = static_cast<Reputation>(i / 4);
      Reputation rep_r = static_cast<Reputation>((i/2) % 2);
      Action act = static_cast<Action>(i % 2);
      ss << "(" << rep_d << "->" << rep_r << "," << act << ") : " << good_probs[i];
      if (i % 4 == 3) { ss << std::endl; }
      else { ss << "\t"; }
    }
    return ss.str();
  }

  bool IsSecondOrder() const {
    if (good_probs[0] == good_probs[4] && good_probs[1] == good_probs[5] &&
        good_probs[2] == good_probs[6] && good_probs[3] == good_probs[7]) { return true; }
    else { return false; }
  }

  std::array<double,8> good_probs;
  // {P(G|B,B,D), P(G|B,B,C), P(G|B,G,D), P(G|B,G,C), P(G|G,B,D), P(G|G,B,C), P(G|G,G,D), P(G|G,G,C)}

  static AssessmentRule MakeDeterministicRule(int id) {
    if (id < 0 || id > 255) {
      throw std::runtime_error("AssessmentRuleDet: id must be between 0 and 255");
    }
    std::array<double,8> good_probs = {0.0};
    for (size_t i = 0; i < 8; i++) {
      if ((id >> i) % 2) { good_probs[i] = 1.0; }
      else { good_probs[i] = 0.0; }
    }
    return AssessmentRule{ good_probs };
  }
};



// Norm is a set of AssessmentRule & ActionRule
class Norm {
  public:
  Norm(const AssessmentRule& rep_d, const ActionRule& act_r) : rd(rep_d.good_probs), ar(act_r.coop_probs) {};
  Norm(const Norm& rhs) : rd(rhs.rd), ar(rhs.ar) {};
  AssessmentRule rd;
  ActionRule ar;
  std::string Inspect() const {
    std::stringstream ss;
    for (int i = 0; i < 4; i++) {
      Reputation X = static_cast<Reputation>(i/2);
      Reputation Y = static_cast<Reputation>(i%2);
      auto p = At(X, Y);
      ss << "(" << X << "->" << Y << "): " << std::get<0>(p) << std::get<1>(p) << ':' << std::get<2>(p);
      ss << ((i % 3 == 2) ? "\n" : "\t");
    }
    return ss.str();
  }
  std::tuple<double,double,double> At(Reputation donor, Reputation recipient) const {
    double c_prob = ar.CProb(donor, recipient);
    double g_prob_c = rd.GProb(donor, recipient, Action::C);
    double g_prob_d = rd.GProb(donor, recipient, Action::D);
    return std::make_tuple(c_prob, g_prob_c, g_prob_d);
  }
  double CProb(Reputation donor, Reputation recipient) const { return ar.CProb(donor, recipient); }
  double GProb(Reputation donor, Reputation recipient, Action act) const { return rd.GProb(donor, recipient, act); }
  bool IsSecondOrder() const {
    return rd.IsSecondOrder() && ar.IsSecondOrder();
  }
  static Norm AllC() { return Norm(AssessmentRule{{1,1,1,1,1,1,1,1}},ActionRule{{1,1,1,1}}); }
  static Norm AllD() { return Norm(AssessmentRule{{0,0,0,0,0,0,0,0}},ActionRule{{0,0,0,0}}); }
};

#endif
