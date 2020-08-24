from global_settings import ccm_raw, icm_raw
import numpy as np
import pandas as pd

ccm_raw.set_index(['permno'], inplace=True)
icm_raw.set_index(['PERMNO'], inplace=True)


def fetch_beta(buy_date, beta_group, p):
    # Denominator
    beta_p = beta_group.loc[[p], :]
    b_mkt = beta_p[(beta_p['DATE'] <= buy_date) & (beta_p.shift(-1)['DATE'] > buy_date)]['b_mkt'].values
    if not len(b_mkt) == 1: raise ValError
    if np.isnan(b_mkt[0]): raise ValError

    return b_mkt


def fetch_eps(buy_date, comp_group, p):
    # Link permno to gvkey
    ccm_p = ccm_raw.loc[[p], :]
    g = ccm_p[(buy_date > ccm_p['linkdt']) & (buy_date <= ccm_p['linkenddt'])]['gvkey'].values
    if not len(g) == 1: raise ValError

    # Numerator
    comp_g = comp_group.loc[[g[0]], :]
    condition = (comp_g['datadate'] <= buy_date) & (comp_g.shift(-1)['datadate'] > buy_date)
    comp_g_0, comp_g_1 = comp_g.shift(-0)[condition], comp_g.shift(-1)[condition]
    comp_g_2, comp_g_3 = comp_g.shift(-2)[condition], comp_g.shift(-3)[condition]
    if not (len(comp_g_0) * len(comp_g_1) * len(comp_g_2) * len(comp_g_3) == 1): raise ValError

    return comp_g_0, comp_g_1, comp_g_2, comp_g_3


# NOTE: eps_func, data_group = [hist, [comp_group]] if ptype == 'hist' else [ibes, [comp_group, ibes_group, adjs]]
#       eps_0_, eps_1_, eps_2_, eps_3_, sic_, gic_, end_date = eps_func(buy_date, data_group, p)
# Above is the right way to use the hist function; delete the comma on the first line if used separately
def hist(buy_date, data_group, p, metric):
    comp_group, = data_group
    comp_g_0, comp_g_1, comp_g_2, comp_g_3 = fetch_eps(buy_date, comp_group, p)

    fyear = lambda comp_g_i: comp_g_i['fyear'].values[0]
    if not fyear(comp_g_3) - fyear(comp_g_2) == 1: raise ValError
    if not fyear(comp_g_2) - fyear(comp_g_1) == 1: raise ValError
    if not fyear(comp_g_1) - fyear(comp_g_0) == 1: raise ValError

    sic_, gic_, end_date = comp_g_0['sic'].values[0], comp_g_0['gind'].values[0], list(comp_g_0['datadate'])[0]
    aje_0_ = comp_g_0['ajex'].values[0]  # WARNING: ajex updated at the end of the previous fiscal year
    comp_0_ = comp_g_0[metric].values[0]  # about 1/4 rows eps == np.nan
    comp_1_ = comp_g_1[metric].values[0] / comp_g_1['ajex'].values[0] * aje_0_  # about 1/4 rows eps == np.nan
    comp_2_ = comp_g_2[metric].values[0] / comp_g_2['ajex'].values[0] * aje_0_  # about 1/4 rows eps == np.nan
    comp_3_ = comp_g_3[metric].values[0] / comp_g_3['ajex'].values[0] * aje_0_  # about 1/4 rows eps == np.nan

    return comp_0_, comp_1_, comp_2_, comp_3_, sic_, gic_, end_date


def extrapolate(buy_date, data_group, p, metric):
    # assert metric in ['epsfx', 'bkvlps'], 'Invalid metric type'
    comp_group, = data_group
    # take part of the fetch_eps function to boost efficiency
    ccm_p = ccm_raw.loc[[p], :]
    g = ccm_p[(buy_date > ccm_p['linkdt']) & (buy_date <= ccm_p['linkenddt'])]['gvkey'].values
    if not len(g) == 1: raise ValError

    comp_g = comp_group.loc[[g[0]], :]
    condition = (comp_g['datadate'] <= buy_date) & (comp_g.shift(-1)['datadate'] > buy_date)
    comp_g_0, comp_g_1 = comp_g.shift(-0)[condition], comp_g.shift(-1)[condition]
    if not (len(comp_g_0) * len(comp_g_1) == 1): raise ValError

    fyear = lambda comp_g_i: comp_g_i['fyear'].values[0]
    if not fyear(comp_g_1) - fyear(comp_g_0) == 1: raise ValError

    sic_, gic_, end_date = comp_g_0['sic'].values[0], comp_g_0['gind'].values[0], list(comp_g_0['datadate'])[0]
    comp_0_ = comp_g_0[metric].values[0]  # about 1/4 rows eps == np.nan
    comp_1_, comp_2_, comp_3_ = comp_0_, comp_0_, comp_0_

    return comp_0_, comp_1_, comp_2_, comp_3_, sic_, gic_, end_date


def ibes(buy_date, data_group, p, metric):
    comp_group, ibes_group, adjs = data_group
    comp_g_0, _, _, _ = fetch_eps(buy_date, comp_group, p)
    sic_, gic_, end_date = comp_g_0['sic'].values[0], comp_g_0['gind'].values[0], list(comp_g_0['datadate'])[0]

    # Link permno to ticker
    icm_p = icm_raw.loc[[p], :]
    t = icm_p[(buy_date > icm_p['sdate']) & (buy_date <= icm_p['edate'])]['TICKER'].values
    if not len(t) == 1: raise ValError

    # Numerator
    ibes_t = ibes_group.loc[[t[0]], :]; adjs_t = adjs.loc[[t[0]], :]
    adjs_t = adjs_t.append(pd.Series(data={'STATPERS': pd.Timestamp(year=2047, month=7, day=1)}, name=t[0]))
    adjs_t = adjs_t[(end_date >= adjs_t['STATPERS']) & (end_date < adjs_t.shift(-1)['STATPERS'])]
    if not len(adjs_t) == 1: raise ValError

    adjs_0_ = adjs_t['ADJSPF'].values[0]  # WARNING: ajex updated at the end of the previous fiscal year
    eps_0_ = comp_g_0[metric].values[0]
    ibes_1_ = ibes_fpi(buy_date, ibes_t, 1) * adjs_0_
    ibes_2_ = ibes_fpi(buy_date, ibes_t, 2) * adjs_0_
    ibes_3_ = ibes_fpi(buy_date, ibes_t, 3) * adjs_0_

    return eps_0_, ibes_1_, ibes_2_, ibes_3_, sic_, gic_, end_date


def ibes_fpi(buy_date, ibes_t, fpi):
    ibes_t_i = ibes_t[ibes_t['FPI'] == fpi]
    ibes_t_i = ibes_t_i[(ibes_t_i['FPEDATS'] > buy_date) & (ibes_t_i['STATPERS'] <= buy_date)]
    if len(ibes_t_i) == 0: raise ValError
    ibes_i_ = ibes_t_i.iloc[[-1], :]['MEDEST'].values[0]

    return ibes_i_


class ValError(Exception):
    pass


# Check date_1, date_2 and date_3 are match
# Check adj_1, adj_2 and adj_3 are match
# Check eps_1, eps_2 and eps_3 are match
