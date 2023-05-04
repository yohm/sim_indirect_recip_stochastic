#%%
import sympy
# %%
r1_ggc, r1_ggd, r1_gbc, r1_gbd, r1_bgc, r1_bgd, r1_bbc, r1_bbd = sympy.symbols('R1_{GGC} R1_{GGD} R1_{GBC} R1_{GBD} R1_{BGC} R1_{BGD} R1_{BBC} R1_{BBD}', nonnegative=True)
r2_ggc, r2_ggd, r2_gbc, r2_gbd, r2_bgc, r2_bgd, r2_bbc, r2_bbd = sympy.symbols('R2_{GGC} R2_{GGD} R2_{GBC} R2_{GBD} R2_{BGC} R2_{BGD} R2_{BBC} R2_{BBD}', nonnegative=True)
p_gg, p_gb, p_bg, p_bb = sympy.symbols('P_{GG} P_{GB} P_{BG} P_{BB}', nonnegative=True)
mu_e = sympy.symbols('\mu_e', positive=True)
mu_a1 = sympy.symbols('\mu_{a1}', positive=True)
mu_a2 = sympy.symbols('\mu_{a2}', positive=True)
# %%
# effective action rules
p_gg_e = (1 - mu_e) * p_gg
p_gb_e = (1 - mu_e) * p_gb
p_bg_e = (1 - mu_e) * p_bg
p_bb_e = (1 - mu_e) * p_bb
p_gg_e
# %%
# effective assessment rules
r1_ggc_e = (1 - mu_a1) * r1_ggc + mu_a1 * (1 - r1_ggc)
r1_ggd_e = (1 - mu_a1) * r1_ggd + mu_a1 * (1 - r1_ggd)
r1_gbc_e = (1 - mu_a1) * r1_gbc + mu_a1 * (1 - r1_gbc)
r1_gbd_e = (1 - mu_a1) * r1_gbd + mu_a1 * (1 - r1_gbd)
r1_bgc_e = (1 - mu_a1) * r1_bgc + mu_a1 * (1 - r1_bgc)
r1_bgd_e = (1 - mu_a1) * r1_bgd + mu_a1 * (1 - r1_bgd)
r1_bbc_e = (1 - mu_a1) * r1_bbc + mu_a1 * (1 - r1_bbc)
r1_bbd_e = (1 - mu_a1) * r1_bbd + mu_a1 * (1 - r1_bbd)
r1_ggc_e
# %%
r2_ggc_e = (1 - mu_a2) * r2_ggc + mu_a2 * (1 - r2_ggc)
r2_ggd_e = (1 - mu_a2) * r2_ggd + mu_a2 * (1 - r2_ggd)
r2_gbc_e = (1 - mu_a2) * r2_gbc + mu_a2 * (1 - r2_gbc)
r2_gbd_e = (1 - mu_a2) * r2_gbd + mu_a2 * (1 - r2_gbd)
r2_bgc_e = (1 - mu_a2) * r2_bgc + mu_a2 * (1 - r2_bgc)
r2_bgd_e = (1 - mu_a2) * r2_bgd + mu_a2 * (1 - r2_bgd)
r2_bbc_e = (1 - mu_a2) * r2_bbc + mu_a2 * (1 - r2_bbc)
r2_bbd_e = (1 - mu_a2) * r2_bbd + mu_a2 * (1 - r2_bbd)
r2_ggc_e

# %%
# sympy.diff(r1_ggc_e, mu_a1)
# %%
r1_bar_gg = p_gg_e * r1_ggc_e + (1 - p_gg_e) * r1_ggd_e
r1_bar_gb = p_gb_e * r1_gbc_e + (1 - p_gb_e) * r1_gbd_e
r1_bar_bg = p_bg_e * r1_bgc_e + (1 - p_bg_e) * r1_bgd_e
r1_bar_bb = p_bb_e * r1_bbc_e + (1 - p_bb_e) * r1_bbd_e
r1_bar_gg
# %%
r2_bar_gg = p_gg_e * r2_ggc_e + (1 - p_gg_e) * r2_ggd_e
r2_bar_gb = p_gb_e * r2_gbc_e + (1 - p_gb_e) * r2_gbd_e
r2_bar_bg = p_bg_e * r2_bgc_e + (1 - p_bg_e) * r2_bgd_e
r2_bar_bb = p_bb_e * r2_bbc_e + (1 - p_bb_e) * r2_bbd_e
r2_bar_bb
# %%
mu = sympy.symbols('\mu', positive=True)

#%%
#r1_bar_gg.subs(mu_e, mu).subs(mu_a1, mu).subs(mu_a2, mu).diff(mu).subs(mu, 0)
# %%
A = r1_bar_gg + r2_bar_gg - r1_bar_gb - r2_bar_gb - r1_bar_bg - r2_bar_bg + r1_bar_bb + r2_bar_bb
A
# %%
B = r1_bar_gb + r2_bar_gb + r1_bar_bg + r2_bar_bg - 2 * (r1_bar_bb + r2_bar_bb) - 2
B

# %%
C = r1_bar_bb + r2_bar_bb
C
# %%
Aprime = A.subs(mu_e, mu).subs(mu_a1, mu).subs(mu_a2, mu).diff(mu).subs(mu, 0)
Aprime
# %%
Bprime = B.subs(mu_e, mu).subs(mu_a1, mu).subs(mu_a2, mu).diff(mu).subs(mu, 0)
Bprime
# %%
Cprime = C.subs(mu_e, mu).subs(mu_a1, mu).subs(mu_a2, mu).diff(mu).subs(mu, 0)
Cprime
# %%
# h_ast = -C/B
h_ast = (-B - sympy.sqrt(B**2 - 4 * A * C)) / (2 * A)
h_ast
# %%
pc = h_ast**2 * p_gg_e + h_ast * (1 - h_ast) * (p_gb_e + p_bg_e) + (1 - h_ast)**2 * p_bb_e
pc

# %%
dh_ast_dmu = h_ast.subs(r1_ggc, 1).subs(r2_ggc, 1).subs(p_gg, 1).subs(mu_e, mu).subs(mu_a1, mu).subs(mu_a2, mu).diff(mu)
dh_ast_dmu

# %%
dpc_dmu = pc.subs(r1_ggc, 1).subs(r2_ggc, 1).subs(p_gg, 1).subs(p_gb, 0).subs(mu_e, mu).subs(mu_a1, mu).subs(mu_a2, mu).diff(mu)
dpc_dmu

# %%
assumptions = sympy.And(
  h_ast > 0, h_ast < 1,
  pc > 0, pc < 1)
assumptions

# %%
with sympy.assuming(assumptions):
  dpc_dmu = pc.subs(r1_ggc, 1).subs(r2_ggc, 1).subs(p_gg, 1).subs(p_gb, 0).subs(p_bg, 1).subs(mu_e, 0).subs(mu_a1, 0).subs(mu_a2, mu).diff(mu).subs(mu, 0)
  # dpc_dmu = pc.subs(r1_ggc, 1).subs(r2_ggc, 1).subs(p_gg, 1).subs(p_gb, 0).subs(mu_e, mu).subs(mu_a1, mu).subs(mu_a2, mu).diff(mu)
dpc_dmu

# %%
assump1 = sympy.And(
  r1_gbd + r2_gbd + r1_bgc + r2_bgc > 2,
  r1_ggd < 1,
  r1_bgc > r1_bgd
)
# this does not finish in reasonable time
# with sympy.assuming(assump1):
#   simplified = dpc_dmu.simplify()
# simplified



# %%
def subs_recip_keep_r2(equation):
  ans = equation.subs( r2_ggc,  1)
  ans = ans.subs(r2_ggd, 1)
  ans = ans.subs(r2_gbc, 0)
  ans = ans.subs(r2_gbd, 0)
  ans = ans.subs(r2_bgc, 1)
  ans = ans.subs(r2_bgd, 1)
  ans = ans.subs(r2_bbc, 0)
  ans = ans.subs(r2_bbd, 0)
  return ans

#%%
def subs_third_r2(equation, r2_ggd_value=1):
  ans = equation.subs( r2_ggc,  1)
  ans = ans.subs(r2_ggd, r2_ggd_value)
  ans = ans.subs(r2_gbd, 1)
  ans = ans.subs(r2_bgc, 1)
  return ans

# %%
def subs_l8_common(equation):
  # substitution of leading eight common prescription
  ans = equation.subs( p_gg,  1)
  ans = ans.subs(r1_ggc, 1)
  ans = ans.subs(r1_ggd, 0)
  ans = ans.subs( p_gb,  0)
  ans = ans.subs(r1_gbd, 1)
  ans = ans.subs( p_bg,  1)
  ans = ans.subs(r1_bgc, 1)
  ans = ans.subs(r1_bgd, 0)
  return ans

# %%
# for the leading eight norms, sensitivity ~ 2*mu_e + mu_a1 + mu_a2
subs_l8_common( subs_recip_keep_r2( pc ) ).subs(mu_a1,0).subs(mu_a2,0).diff(mu_e).subs(mu_e, 0).simplify()   #=> -2
# %%
subs_l8_common( subs_recip_keep_r2( pc ) ).subs(mu_e,0).subs(mu_a2,0).diff(mu_a1).subs(mu_a1, 0).simplify()   #=> -1
# %%
subs_l8_common( subs_recip_keep_r2( pc ) ).subs(mu_e,0).subs(mu_a1,0).diff(mu_a2).subs(mu_a2, 0).simplify()   #=> -1

# %%
subs_l8_common( subs_recip_keep_r2( pc ) ).subs(mu_e,mu).subs(mu_a1,mu).subs(mu_a2,mu).diff(mu).subs(mu, 0).simplify()  #=> -4

# %%
subs_third_r2( subs_l8_common(pc), 1).subs(mu_e,mu).subs(mu_a1,mu).subs(mu_a2,mu).diff(mu).subs(mu, 0).simplify()  #=> -5/2 (-3/2 mu_e - 1/2 mu_a1 - 1/2 mu_a2)
# %%
subs_third_r2( subs_l8_common(pc), 0).subs(mu_e,mu).subs(mu_a1,mu).subs(mu_a2,mu).diff(mu).subs(mu, 0).simplify()  #=> -3  (-2 mu_e - 1/2 mu_a1 - 1/2 mu_a2)


# %%
def subs_norm(equation, norm):
  # format of list
  # row: (GG) (GB) (BG) (BB)
  # column: (P R1C R1D R2C R2D)
  ans = equation.subs( p_gg,  norm[0][0])
  ans = ans.subs(r1_ggc, norm[0][1])
  ans = ans.subs(r1_ggd, norm[0][2])
  ans = ans.subs(r2_ggc, norm[0][3])
  ans = ans.subs(r2_ggd, norm[0][4])
  ans = ans.subs( p_gb,  norm[1][0])
  ans = ans.subs(r1_gbc, norm[1][1])
  ans = ans.subs(r1_gbd, norm[1][2])
  ans = ans.subs(r2_gbc, norm[1][3])
  ans = ans.subs(r2_gbd, norm[1][4])
  ans = ans.subs( p_bg,  norm[2][0])
  ans = ans.subs(r1_bgc, norm[2][1])
  ans = ans.subs(r1_bgd, norm[2][2])
  ans = ans.subs(r2_bgc, norm[2][3])
  ans = ans.subs(r2_bgd, norm[2][4])
  ans = ans.subs( p_bb,  norm[3][0])
  ans = ans.subs(r1_bbc, norm[3][1])
  ans = ans.subs(r1_bbd, norm[3][2])
  ans = ans.subs(r2_bbc, norm[3][3])
  ans = ans.subs(r2_bbd, norm[3][4])
  return ans

# %%
l1_norm = [
  [1, 1, 0, 1, 1],
  [0, 1, 1, 0, 0],
  [1, 1, 0, 1, 1],
  [1, 1, 0, 0, 0]
]

# %%
subs_norm( dpc_dmu, l1_norm).limit(mu, 0)
