import numpy as np

rf = 0.020  # risk free rate
rp = 0.090  # risk premium
tg = 0.050  # terminal growth rate


def DCF(eps_0, eps_1, eps_2, eps_3, pct, beta):
    eeps_0, eeps_1, eeps_2 = (1-pct)*eps_0 + pct*eps_1, (1-pct)*eps_1 + pct*eps_2, (1-pct)*eps_2 + pct*eps_3
    discount = rf + beta * rp
    pv = np.divide(eeps_0, 1+discount) + np.divide(eeps_1, (1+discount)**2) + np.divide(eeps_2, (1+discount)**3) + \
         np.divide(eeps_2*(1+tg), ((1+discount)**3)*(discount-tg+1e-8))
    return pv


def Multiple(comp_1, comp_2, pct, median_multiple):
    forward_comp = (1-pct)*comp_1 + pct*comp_2
    pv = forward_comp * median_multiple
    return pv
