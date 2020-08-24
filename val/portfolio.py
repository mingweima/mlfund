from val.vutils.global_settings import CONFIG_FOLDER, groups, PERMNO, SIC1, SIC2, GIC1, GIC2
from val.vutils.load_data import construct_daily, load_beta, load_ibes, load_comp, load_adjs, load_multiple, load_partial_dsf
from val.val_data import fetch_beta, hist, ibes, extrapolate, ValError
from val.val_model import DCF, Multiple
import numpy as np
import json
import os
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
from datetime import datetime

with open(os.path.join(CONFIG_FOLDER, 'portfolio.json'), 'rb') as handle:
    params = json.load(handle)
    drop = params['drop']
    keep = params['keep']
    inds = params['inds']


def group_proc(mtype, buy_date, ptype, group):
    permno_group = [_ for _ in PERMNO if _[:2] == group]
    if mtype == 'dcf':
        permno_group, sic_group, gic_group, pv_group = dcf_valuation(buy_date, permno_group, group, ptype)
    else:
        permno_group, sic_group, gic_group, pv_group = multi_valuation(buy_date, permno_group, group, ptype, mtype)
    return permno_group, sic_group, gic_group, pv_group


def construct_portfolio(buy_date, ptype, mtype, conn, multip=16):
    assert inds in ['none', 'sic1', 'sic2', 'gic1', 'gic2'], 'Invalid industrial type'
    assert ptype in ['hist', 'ibes', 'extrapolate'], 'Invalid predictor type'
    assert mtype in ['dcf', 'PE', 'PB'], 'Invalid model type'

    # full permno from ccm
    permno, sic, gic, pv = np.array([]), np.array([]), np.array([]), np.array([])

    if multip == 1:
        for group in tqdm(groups):
            permno_group, sic_group, gic_group, pv_group = group_proc(mtype, buy_date, ptype, group)
            permno = np.concatenate([permno, permno_group])
            sic = np.concatenate([sic, sic_group])
            gic = np.concatenate([gic, gic_group])
            pv = np.concatenate([pv, pv_group])

    else:
        print(f'{datetime.now()} Started multi-processing for buy date {buy_date}!')
        pool = Pool(multip)
        partial_group_proc = partial(group_proc, mtype, buy_date, ptype)
        results = pool.map(partial_group_proc, groups)
        pool.close()
        pool.join()
        for result in results:
            permno_group, sic_group, gic_group, pv_group = result
            permno = np.concatenate([permno, permno_group])
            sic = np.concatenate([sic, sic_group])
            gic = np.concatenate([gic, gic_group])
            pv = np.concatenate([pv, pv_group])
        print(f'{datetime.now()} Finished multi-processing for buy date {buy_date}!')
    # permno with well-defined return (unadjusted)
    daily_df = construct_daily(buy_date, list(permno), conn=conn)
    arg = np.array([list(permno).index(_) for _ in list(daily_df['permno'])]).astype(int)
    permno, sic, gic, pv, prc = permno[arg], sic[arg], gic[arg], pv[arg], np.array(daily_df['prc'])
    norm_pv = np.divide(pv, prc)

    # permno long/short
    ls_func = industrial_ls if not inds == 'none' else base_ls
    (long_permno, long_sic, long_gic), (short_permno, short_sic, short_gic) = ls_func(norm_pv, permno, sic, gic)
    long_permno, short_permno = long_permno.tolist(), short_permno.tolist()
    long_sic, short_sic = long_sic.tolist(), short_sic.tolist()
    long_gic, short_gic = long_gic.tolist(), short_gic.tolist()

    return (long_permno, long_sic, long_gic), (short_permno, short_sic, short_gic)


def dcf_valuation(buy_date, permno_group, group, ptype):
    beta_group, comp_group, ibes_group, adjs = _load_group_data(group)
    eps_0, eps_1, eps_2, eps_3 = np.array([]), np.array([]), np.array([]), np.array([])
    pct, beta, sic, gic, permno = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    metric = 'epsfx'
    for p in permno_group:
        try:
            if ptype == 'hist':
                eps_func, data_group = [hist, [comp_group]]
            elif ptype == 'extrapolate':
                eps_func, data_group = [extrapolate, [comp_group]]
            else:  # ptype == 'ibes'
                eps_func, data_group = [ibes, [comp_group, ibes_group, adjs]]
            eps_0_, eps_1_, eps_2_, eps_3_, sic_, gic_, end_date = eps_func(buy_date, data_group, p, metric)
            b_mkt = fetch_beta(buy_date, beta_group, p)

            if not (np.isnan(eps_0_) or np.isnan(eps_1_) or np.isnan(eps_2_) or np.isnan(eps_3_)):
                eps_0 = np.append(eps_0, eps_0_); eps_1 = np.append(eps_1, eps_1_)
                eps_2 = np.append(eps_2, eps_2_); eps_3 = np.append(eps_3, eps_3_)
                pct, beta = np.append(pct, int((buy_date - end_date).days) / 365), np.append(beta, b_mkt[0])
                sic, gic, permno = np.append(sic, sic_), np.append(gic, gic_), np.append(permno, p)

        except (KeyError, ValError) as Error:
            pass

    pv = DCF(eps_0, eps_1, eps_2, eps_3, pct, beta)

    return permno, sic, gic, pv


def multi_valuation(buy_date, permno_group, group, ptype, mtype):
    assert mtype in ['PE', 'PB'], 'Invalid multiple type'

    beta_group, comp_group, ibes_group, adjs = _load_group_data(group)
    comp_1, comp_2 = np.array([]), np.array([])
    pct, median_multiple, sic, gic, permno = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    multiple_df = load_multiple(mtype).set_index(['SICCD'], inplace=False)
    dsf_group = load_partial_dsf(group).set_index(['PERMNO'], inplace=False)

    for p in permno_group:
        try:
            if (ptype == 'hist') & (mtype == 'PE'):
                metric = 'epsfx'
                comp_func, data_group = [hist, [comp_group]]
            elif (ptype == 'hist') & (mtype == 'PB'):
                metric = 'bkvlps'
                comp_func, data_group = [hist, [comp_group]]
            elif (ptype == 'extrapolate') & (mtype == 'PE'):
                metric = 'epsfx'
                comp_func, data_group = [extrapolate, [comp_group]]
            elif (ptype == 'extrapolate') & (mtype == 'PB'):
                metric = 'bkvlps'
                comp_func, data_group = [extrapolate, [comp_group]]
            elif (ptype == 'ibes') & (mtype == 'PE'):
                metric = 'epsfx'
                comp_func, data_group = [ibes, [comp_group, ibes_group, adjs]]
            else:  #dtype == 'ibes' & mtype =='PB'
                metric = 'bkvlps'
                comp_func, data_group = [ibes, [comp_group, ibes_group, adjs]]
            comp_0_, comp_1_, comp_2_, comp_3_, sic_, gic_, end_date = comp_func(buy_date, data_group, p, metric)
            # note that for a PERMNO, the SICCD code changes over time, so need to apply the date_filt first
            sic_ = dsf_group.loc[dsf_group['date'] == buy_date].loc[p, 'SICCD']
            # note that this sic_ is actually the CRSP SICCD code
            date_filt = multiple_df['date'] == buy_date
            # KeyError presented if .loc[sic_] could not find the series
            median_variable = multiple_df.loc[date_filt].loc[sic_]['median_' + mtype]

            if not (np.isnan(comp_1_) or np.isnan(comp_2_)):
                median_multiple = np.append(median_multiple, median_variable)
                comp_1 = np.append(comp_1, comp_1_); comp_2 = np.append(comp_2, comp_2_)
                pct = np.append(pct, int((buy_date - end_date).days) / 365)
                sic, gic, permno = np.append(sic, sic_), np.append(gic, gic_), np.append(permno, p)
        except (KeyError, ValError) as Error:
            pass

    pv = Multiple(comp_1, comp_2, pct, median_multiple)

    return permno, sic, gic, pv


def base_ls(norm_pv, permno, sic, gic):
    args, drop_len = np.argsort(norm_pv), int(len(permno) * drop)
    args = args[drop_len: -drop_len] if drop_len != 0 else args
    keep_len = int(len(args) * keep)

    if keep_len != 0:
        long_permno, short_permno = permno[args[-keep_len:]], permno[args[:keep_len]]
        long_sic, short_sic = sic[args[-keep_len:]], sic[args[:keep_len]]
        long_gic, short_gic = gic[args[-keep_len:]], gic[args[:keep_len]]
    else:
        long_permno, short_permno = np.array([]), np.array([])
        long_sic, short_sic = np.array([]), np.array([])
        long_gic, short_gic = np.array([]), np.array([])

    return (long_permno, long_sic, long_gic), (short_permno, short_sic, short_gic)


def industrial_ls(norm_pv, permno, sic, gic):
    long_permno, short_permno = np.array([]), np.array([])
    long_sic, short_sic = np.array([]), np.array([])
    long_gic, short_gic = np.array([]), np.array([])

    if inds == 'sic1':
        (cat, init) = (SIC1, sic.astype('<U1'))
    elif inds == 'sic2':
        (cat, init) = (SIC2, sic.astype('<U2'))
    elif inds == 'gic1':
        (cat, init) = (GIC1, gic.astype('<U4'))
    else:
        (cat, init) = (GIC2, gic.astype('<U6'))

    for _ in cat:
        permno_, norm_pv_, sic_, gic_ = permno[init == _], norm_pv[init == _], sic[init == _], gic[init == _]
        zip_long_, zip_short_ = base_ls(norm_pv_, permno_, sic_, gic_)
        (long_permno_, long_sic_, long_gic_), (short_permno_, short_sic_, short_gic_) = zip_long_, zip_short_
        long_permno = np.append(long_permno, long_permno_); short_permno = np.append(short_permno, short_permno_)
        long_sic = np.append(long_sic, long_sic_); short_sic = np.append(short_sic, short_sic_)
        long_gic = np.append(long_gic, long_gic_); short_gic = np.append(short_gic, short_gic_)

    return (long_permno, long_sic, long_gic), (short_permno, short_sic, short_gic)


def _load_group_data(group):
    beta_group = load_beta(group).set_index(['PERMNO'])
    comp_group = load_comp(group).set_index(['gvkey'])
    ibes_group = load_ibes(group).set_index(['TICKER'])
    adjs = load_adjs().set_index(['TICKER'])

    return beta_group, comp_group, ibes_group, adjs


# from global_settings import sp500_full
# def construct_portfolio(business_day, lag=5):
#     # permno = ['84788', '89393', '78877', '53613'] # AMAZON, NETFLIX, CHESAPEAKE, MICRON
#     permno = ['14593', '90319', '13407', '10107']  # APPLE, GOOGLE, FACEBOOK, MICROSOFT
#     idx = list(sp500['Date']).index(business_day)
#     business_lag = sp500['Date'][idx - lag]
#     recent = []
#
#     for p in permno:
#         daily_df_0 = construct_daily(business_day, p)
#         daily_df_l = construct_daily(business_lag, p)
#         adjprc_0 = float(daily_df_0['prc']) / float(daily_df_0['cfacpr'])
#         adjprc_l = float(daily_df_l['prc']) / float(daily_df_l['cfacpr'])
#         recent.append(adjprc_0/adjprc_l - 1)
#
#     short_permno = list(np.array(permno)[np.array(recent).argsort()[:len(permno)//2]])
#     long_permno = list(np.array(permno)[np.array(recent).argsort()[len(permno)//2:]])
#
#     return long_permno, short_permno
