#ifndef NORM_HPP
#define NORM_HPP

#include <iostream>
#include <iomanip>
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

  std::array<double,4> coop_probs; // P(C|B,B), P(C|B,G), P(C|G,B), P(C|G,G)

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
    ss << "ActionRule: " << ID() << std::endl;
    for (size_t i = 0; i < 4; i++) {
      Reputation rep_d = static_cast<Reputation>(i / 2);
      Reputation rep_r = static_cast<Reputation>(i % 2);
      ss << "(" << rep_d << "->" << rep_r << ") : " << coop_probs[i];
      if (i % 2 == 1) { ss << std::endl; }
      else { ss << "\t"; }
    }
    return ss.str();
  }

  ActionRule RescaleWithError(double mu_e) const {
    std::array<double,4> rescaled = {0.0};
    for (size_t i = 0; i < 4; i++) {
      rescaled[i] = (1.0 - mu_e) * coop_probs[i];
    }
    return ActionRule{rescaled};
  }

  bool IsSecondOrder() const {
    if (coop_probs[0] == coop_probs[2] && coop_probs[1] == coop_probs[3]) { return true; }
    else { return false; }
  }

  bool IsDeterministic() const {
    if (( coop_probs[0] == 0.0 || coop_probs[0] == 1.0 ) &&
        ( coop_probs[1] == 0.0 || coop_probs[1] == 1.0 ) &&
        ( coop_probs[2] == 0.0 || coop_probs[2] == 1.0 ) &&
        ( coop_probs[3] == 0.0 || coop_probs[3] == 1.0 ) ) {
      return true;
    }
    return false;
  }

  int ID() const {
    if (!IsDeterministic()) { return -1; }
    int id = 0;
    for (size_t i = 0; i < 4; i++) {
      if (coop_probs[i] == 1.0) { id += 1 << i; }
    }
    return id;
  }

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

  static ActionRule DISC() {
    return ActionRule{ {0.0, 1.0, 0.0, 1.0} };
  }
  static ActionRule ALLC() {
    return ActionRule{ {1.0, 1.0, 1.0, 1.0} };
  }
  static ActionRule ALLD() {
    return ActionRule{ {0.0, 0.0, 0.0, 0.0} };
  }
};

bool operator==(const ActionRule& t1, const ActionRule& t2) {
  return t1.coop_probs[0] == t2.coop_probs[0] &&
         t1.coop_probs[1] == t2.coop_probs[1] &&
         t1.coop_probs[2] == t2.coop_probs[2] &&
         t1.coop_probs[3] == t2.coop_probs[3];
}
bool operator!=(const ActionRule& t1, const ActionRule& t2) { return !(t1 == t2); }

class AssessmentRule {
  public:
  AssessmentRule(const std::array<double,8>& g_probs) : good_probs(g_probs) {};
  // {P(G|B,B,D), P(G|B,B,C), P(G|B,G,D), P(G|B,G,C), P(G|G,B,D), P(G|G,B,C), P(G|G,G,D), P(G|G,G,C)}
  std::array<double,8> good_probs;

  AssessmentRule Clone() const { return AssessmentRule(good_probs); }
  AssessmentRule SwapGB() const {
    std::array<double,8> new_good_probs = {1.0-good_probs[6], 1.0-good_probs[7], 1.0-good_probs[4], 1.0-good_probs[5],
                                           1.0-good_probs[2], 1.0-good_probs[3], 1.0-good_probs[0], 1.0-good_probs[1]};
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
    ss << "AssessmentRule: " << ID() << std::endl;
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

  bool IsDeterministic() const {
    if (( good_probs[0] == 0.0 || good_probs[0] == 1.0 ) &&
        ( good_probs[1] == 0.0 || good_probs[1] == 1.0 ) &&
        ( good_probs[2] == 0.0 || good_probs[2] == 1.0 ) &&
        ( good_probs[3] == 0.0 || good_probs[3] == 1.0 ) &&
        ( good_probs[4] == 0.0 || good_probs[4] == 1.0 ) &&
        ( good_probs[5] == 0.0 || good_probs[5] == 1.0 ) &&
        ( good_probs[6] == 0.0 || good_probs[6] == 1.0 ) &&
        ( good_probs[7] == 0.0 || good_probs[7] == 1.0 ) ) {
      return true;
    }
    return false;
  }

  int ID() const {
    if (!IsDeterministic()) { return -1; }
    int id = 0;
    for (size_t i = 0; i < 8; i++) {
      if (good_probs[i] == 1.0) { id += 1 << i; }
    }
    return id;
  }

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

  static AssessmentRule AllGood() {
    return AssessmentRule::MakeDeterministicRule(0b11111111);
  }
  static AssessmentRule AllBad() {
    return AssessmentRule::MakeDeterministicRule(0b00000000);
  }
  static AssessmentRule ImageScoring() {
    return AssessmentRule{{0,1,0,1,0,1,0,1}};
  }
  static AssessmentRule KeepRecipient() {
    return AssessmentRule{{0,0,1,1,0,0,1,1}};
  }
};

bool operator==(const AssessmentRule& t1, const AssessmentRule& t2) {
  return t1.good_probs[0] == t2.good_probs[0] &&
         t1.good_probs[1] == t2.good_probs[1] &&
         t1.good_probs[2] == t2.good_probs[2] &&
         t1.good_probs[3] == t2.good_probs[3] &&
         t1.good_probs[4] == t2.good_probs[4] &&
         t1.good_probs[5] == t2.good_probs[5] &&
         t1.good_probs[6] == t2.good_probs[6] &&
         t1.good_probs[7] == t2.good_probs[7];
}
bool operator!=(const AssessmentRule& t1, const AssessmentRule& t2) {
  return !(t1 == t2);
}

// Norm is a set of AssessmentRule & ActionRule
class Norm {
  public:
  Norm(const AssessmentRule& R_d, const AssessmentRule& R_r, const ActionRule& act)
  : Rd(R_d.good_probs), Rr(R_r.good_probs), P(act.coop_probs) {};
  Norm(const Norm& rhs) : Rd(rhs.Rd), Rr(rhs.Rr), P(rhs.P) {};
  AssessmentRule Rd, Rr;  // assessment rules to assess donor and recipient
  ActionRule P;  // action rule
  std::string Inspect() const {
    std::stringstream ss;
    if (IsDeterministic()) {
      ss << "Norm: 0x" << std::hex << ID() << std::endl;
      for (int i = 3; i >= 0; i--) {
        Reputation X = static_cast<Reputation>(i / 2);
        Reputation Y = static_cast<Reputation>(i % 2);
        Action A = (CProb(X,Y) == 1.0) ? Action::C : Action::D;
        Action notA = FlipAction(A);
        Reputation donor_rep = Rd.GProb(X, Y, A) == 1.0 ? Reputation::G : Reputation::B;
        Reputation donor_rep_not = Rd.GProb(X, Y, notA) == 1.0 ? Reputation::G : Reputation::B;
        Reputation recip_rep = Rr.GProb(X, Y, A) == 1.0 ? Reputation::G : Reputation::B;
        Reputation recip_rep_not = Rr.GProb(X, Y, notA) == 1.0 ? Reputation::G : Reputation::B;
        ss << "(" << X << "->" << Y << "):" << A << donor_rep << donor_rep_not << ":" << recip_rep << recip_rep_not << "\t";
      }
      ss << "\n";
    }
    for (int i = 3; i >= 0; i--) {
      Reputation X = static_cast<Reputation>(i/2);
      Reputation Y = static_cast<Reputation>(i%2);
      double c_prob = P.CProb(X, Y);
      double gd_prob_d = Rd.GProb(X, Y, Action::D);
      double gd_prob_c = Rd.GProb(X, Y, Action::C);
      double gr_prob_d = Rr.GProb(X, Y, Action::D);
      double gr_prob_c = Rr.GProb(X, Y, Action::C);
      ss << std::setprecision(3) << std::fixed;
      ss << "(" << X << "->" << Y << "): "
          << "c_prob:" << c_prob << " : "
          << "donor_gprob (c:" << gd_prob_c << ",d:" << gd_prob_d << ") : "
          << "recip_gprob (c:" << gr_prob_c << ",d:" << gr_prob_d << ")\n";
    }
    return ss.str();
  }
  double CProb(Reputation donor, Reputation recipient) const { return P.CProb(donor, recipient); }
  double GProbDonor(Reputation donor, Reputation recipient, Action act) const { return Rd.GProb(donor, recipient, act); }
  double GProbRecip(Reputation donor, Reputation recipient, Action act) const { return Rr.GProb(donor, recipient, act); }
  Norm SwapGB() const {
    return Norm(Rd.SwapGB(), Rr.SwapGB(), P.SwapGB());
  }
  bool IsSecondOrder() const {
    return P.IsSecondOrder() && Rd.IsSecondOrder() && Rr.IsSecondOrder();
  }
  bool IsDeterministic() const {
    return P.IsDeterministic() && Rd.IsDeterministic() && Rr.IsDeterministic();
  }
  bool IsRecipKeep() const { // recipient's reputation is kept
    return Rr.IsDeterministic() && Rr.ID() == 0b11001100;
  }
  Norm RescaleWithError(double mu_e, double mu_a_donor, double mu_a_recip) const { // return the norm that takes into account error probabilities
    std::array<double,8> good_probs_donor = {0.0};
    std::array<double,8> good_probs_recip = {0.0};
    for (size_t i = 0; i < 8; i++) {
      good_probs_donor[i] = (1.0 - 2.0 * mu_a_donor) * Rd.good_probs[i] + mu_a_donor;
      good_probs_recip[i] = (1.0 - 2.0 * mu_a_recip) * Rr.good_probs[i] + mu_a_recip;
    }
    return Norm{ {good_probs_donor}, {good_probs_recip}, P.RescaleWithError(mu_e) };
  }
  int ID() const {
    if (!IsDeterministic()) { return -1; }
    int id = 0;
    id += Rd.ID() << 12;
    id += Rr.ID() << 4;
    id += P.ID();
    return id;
  }
  static Norm ConstructFromID(int id) {
    if (id < 0 || id >= (1<<20)) {
      throw std::runtime_error("Norm: id must be between 0 and 2^20-1");
    }
    int Rd_id = (id >> 12) & 0xFF;
    int Rr_id = (id >> 4) & 0xFF;
    int P_id = id & 0xF;
    return Norm(AssessmentRule::MakeDeterministicRule(Rd_id),
                AssessmentRule::MakeDeterministicRule(Rr_id),
                ActionRule::MakeDeterministicRule(P_id));
  }
  static Norm AllC() { return Norm(AssessmentRule{{1,1,1,1,1,1,1,1}}, AssessmentRule{{1,1,1,1,1,1,1,1}}, ActionRule{{1,1,1,1}}); }
  static Norm AllD() { return Norm(AssessmentRule{{0,0,0,0,0,0,0,0}}, AssessmentRule{{0,0,0,0,0,0,0,0}}, ActionRule{{0,0,0,0}}); }
  static Norm ImageScoring() {
    return Norm( AssessmentRule::ImageScoring(),
                 AssessmentRule::KeepRecipient(),
                 ActionRule::DISC() );
  }
  static Norm L1() {
    return Norm({{0,1,0,1,1,1,0,1}},
                AssessmentRule::KeepRecipient(),
                {{1,1,0,1}});
  }
  static Norm L2() {
    return Norm({{0,1,0,1,1,0,0,1}},
                AssessmentRule::KeepRecipient(),
                {{1,1,0,1}});
  }
  static Norm L3() {
    return Norm({{1,1,0,1,1,1,0,1}},
                AssessmentRule::KeepRecipient(),
                {{0,1,0,1}});
  }
  static Norm L4() {
    return Norm({{1,0,0,1,1,1,0,1}},
                AssessmentRule::KeepRecipient(),
                {{0,1,0,1}});
  }
  static Norm L5() {
    return Norm({{1,1,0,1,1,0,0,1}},
                AssessmentRule::KeepRecipient(),
                {{0,1,0,1}});
  }
  static Norm L6() {
    return Norm({{1,0,0,1,1,0,0,1}},
                AssessmentRule::KeepRecipient(),
                {{0,1,0,1}});
  }
  static Norm L7() {
    return Norm({{0,0,0,1,1,1,0,1}},
                AssessmentRule::KeepRecipient(),
                {{0,1,0,1}});
  }
  static Norm L8() {
    return Norm({{0,0,0,1,1,0,0,1}},
                AssessmentRule::KeepRecipient(),
                {{0,1,0,1}});
  }
};

bool operator==(const Norm& n1, const Norm& n2) {
  return n1.P == n2.P && n1.Rd == n2.Rd && n1.Rr == n2.Rr;
}
bool operator!=(const Norm& n1, const Norm& n2) {
  return !(n1 == n2);
}

#endif
