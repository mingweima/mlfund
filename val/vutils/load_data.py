import pandas as pd
import os

from pandas.errors import DtypeWarning

from val.vutils.global_settings import DATA_FOLDER
import warnings


# from global_settings import icm_raw, ccm, groups


def construct_daily(date, permno, conn):
    permno = check(permno)
    daily_df = conn.raw_sql(f"""
                            select a.date, a.permno, b.ticker, b.shrcd, b.siccd, a.ret, 
                            abs(a.prc) as prc, a.shrout, a.cfacpr, a.cfacshr
                            from crsp.dsf as a
                            left join crsp.msenames as b
                            on a.permno = b.permno
                            and b.namedt <= a.date
                            and a.date <= b.nameendt
                            and a.date = '{date}'
                            where b.permno in {permno if len(permno) > 1 else '(' + permno[0] + ')'}
                            """)
    daily_df = clean_crsp(daily_df)

    return daily_df


def check(permno):
    assert isinstance(permno, (list, str)), 'invalid permno data type'
    if isinstance(permno, list):
        assert len(permno) != 0, 'zero permno list length'
    if isinstance(permno, str):
        permno = list([permno])
    permno = tuple(permno)

    return permno


def clean_crsp(daily_df):
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    daily_df['permno'] = daily_df['permno'].astype(int).astype(str)
    daily_df = daily_df[daily_df['siccd'].notna()]
    daily_df['siccd'] = daily_df['siccd'].astype(int).astype(str).apply(lambda _: _.zfill(4))
    # drop abnormal cfacpr and cfacshr != 0.0
    notzero = (daily_df['cfacpr'] != 0.0) & (daily_df['cfacpr'] != 0.0)
    daily_df = daily_df.loc[notzero, :].reset_index(drop=True, inplace=False)
    # drop trading suspension
    notnull = (daily_df['ret'].notnull()) & (daily_df['prc'].notnull())
    daily_df = daily_df.loc[notnull, :]

    daily_df.reset_index(drop=True, inplace=False)

    return daily_df


def load_beta(group):
    file_name = 'beta' if group == '' else '_'.join(['beta', group])
    beta_group = pd.read_csv(os.path.join(DATA_FOLDER, 'beta', file_name + '.csv'))
    beta_group['PERMNO'] = beta_group['PERMNO'].astype(str)
    beta_group['DATE'] = pd.to_datetime(beta_group['DATE'].astype(str), format='%Y%m%d')
    beta_group.sort_values(['PERMNO', 'DATE'], ascending=[True, True], inplace=True)
    beta_group.reset_index(drop=True, inplace=True)

    return beta_group


def load_comp(group):
    file_name = 'comp' if group == '' else '_'.join(['comp', group])
    comp_group = pd.read_csv(os.path.join(DATA_FOLDER, 'comp', file_name) + '.csv')
    comp_group['datadate'] = pd.to_datetime(comp_group['datadate'].astype(str), format='%Y%m%d')
    comp_group['gvkey'] = comp_group['gvkey'].astype(int).astype(str).apply(lambda _: _.zfill(6))
    comp_group['sic'] = comp_group['sic'].astype(int).astype(str).apply(lambda _: _.zfill(4))
    gic_filter = comp_group['gind'].notna()
    comp_group.loc[gic_filter, 'gind'] = comp_group.loc[gic_filter, 'gind'].astype(int).astype(str)

    comp_group = comp_group.loc[comp_group['fyear'].notna()]
    comp_group = comp_group.loc[comp_group['indfmt'] == 'INDL']
    comp_group = comp_group.loc[comp_group['ajex'] != 0, :]  # drop abnormal ajex == 0 rows
    comp_group.sort_values(['gvkey', 'datadate'], ascending=[True, True], inplace=True)
    comp_group.reset_index(drop=True, inplace=True)

    return comp_group


def load_ibes(group):
    file_name = 'ibes' if group == '' else '_'.join(['ibes', group])
    ibes_group = pd.read_csv(os.path.join(DATA_FOLDER, 'ibes', file_name + '.csv'))
    ibes_group['STATPERS'] = pd.to_datetime(ibes_group['STATPERS'].astype(str), format='%Y%m%d')
    ibes_group['FPEDATS'] = pd.to_datetime(ibes_group['FPEDATS'].astype(str), format='%Y%m%d')
    ibes_group = ibes_group[ibes_group['FPEDATS'] > ibes_group['STATPERS']]
    ibes_group.sort_values(['TICKER', 'FPI', 'STATPERS'], ascending=[True, True, True], inplace=True)
    ibes_group.reset_index(drop=True, inplace=True)

    return ibes_group


def load_adjs():
    adjs = pd.read_csv(os.path.join(DATA_FOLDER, 'ibes', 'adjs.csv'))
    adjs['STATPERS'] = pd.to_datetime(adjs['STATPERS'].astype(str), format='%Y%m%d')
    adjs.sort_values(['TICKER', 'STATPERS'], ascending=[True, True], inplace=True)
    adjs.reset_index(drop=True, inplace=True)

    return adjs


# group: str, range from 10 to 93, '' for the whole file
def load_dsf(group):
    file_name = 'dsf' if group == '' else '_'.join(['dsf', group])
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=DtypeWarning)
        dsf_group = pd.read_csv(os.path.join(DATA_FOLDER, 'dsf', file_name + '.csv'))
    dsf_group['date'] = pd.to_datetime(dsf_group['date'].astype(str), format='%Y%m%d')
    dsf_group['PERMNO'] = dsf_group['PERMNO'].astype(str)
    dsf_group.dropna(axis='index', how='any', inplace=True)
    # drop the 'Z's in the SICCD column, and dealing with the mixed datatype issue
    # Handle the annoying futures warning
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        dsf_group = dsf_group.loc[dsf_group['SICCD'] != 'Z', :]  # returns False, warning is suppressed
        dsf_group = dsf_group.loc[dsf_group['RET'] != 'C', :]
    dsf_group['SICCD'] = dsf_group['SICCD'].astype(float).astype(int).astype(str).apply(lambda _: _.zfill(4))
    dsf_group['HSICCD'] = dsf_group['HSICCD'].astype(str)
    # drop the 'C's in the RET column, before converting str to float
    dsf_group['RET'] = dsf_group['RET'].astype(float)
    dsf_group.sort_values(['PERMNO', 'date'], ascending=[True, True], inplace=True)
    dsf_group.reset_index(drop=True, inplace=True)

    return dsf_group


# To make the multi_valuation more efficient
def load_partial_dsf(group):
    file_name = 'dsf' if group == '' else '_'.join(['dsf', group])
    dsf_group = pd.read_csv(os.path.join(DATA_FOLDER, 'partial_dsf', file_name + '.csv'))
    dsf_group['date'] = pd.to_datetime(dsf_group['date'].astype(str), format='%Y-%m-%d')
    dsf_group['PERMNO'] = dsf_group['PERMNO'].astype(str)
    dsf_group['SICCD'] = dsf_group['SICCD'].astype(str)

    return dsf_group


# ex. load_multiple('PE') to load the industry median PE dataframe
def load_multiple(multiple):
    multiple_df = pd.read_csv(os.path.join(DATA_FOLDER, 'multiples', multiple + '.csv'))
    multiple_df['SICCD'] = multiple_df['SICCD'].astype(str)
    multiple_df['date'] = pd.to_datetime(multiple_df['date'].astype(str), format='%Y-%m-%d')
    # temporarily discard negative PE values for valuation purposes
    multiple_df = multiple_df.loc[multiple_df['median_' + multiple] >= 0]
    multiple_df.reset_index(drop=True, inplace=True)

    return multiple_df
# Split functions
# def split_dsf():
#     dsf = pd.read_csv(r'E:\Git\multiples\dsf.csv')
#     for group in tqdm(groups):
#         dsf_group = dsf[dsf['PERMNO'].astype(str).apply(lambda _: _[:2] == group)]
#         dsf_group.reset_index(drop=True, inplace=True)
#         dsf_group.to_csv(os.path.join(FOLDER, '_'.join(['dsf', group]) + '.csv'), index=False)
#
# def split_beta():
#     beta = pd.read_csv(os.path.join(DATA_FOLDER, 'beta', 'beta.csv'))
#     for group in groups:
#         beta_group = beta[beta['PERMNO'].astype(str).apply(lambda _: _[:2] == group)]
#         beta_group.reset_index(drop=True, inplace=True)
#         beta_group.to_csv(os.path.join(DATA_FOLDER, 'beta', '_'.join(['beta', group + '.csv'])), index=False)
#
#
# def split_comp():
#     comp = pd.read_csv(os.path.join(DATA_FOLDER, 'comp', 'comp.csv'))
#     for group in groups:
#         ccm_group = ccm_raw[ccm_raw['permno'].apply(lambda _: _[:2] == group)]
#         comp_group = comp[comp['gvkey'].isin(list(set(ccm_group['gvkey'])))]
#         comp_group.reset_index(drop=True, inplace=True)
#         comp_group.to_csv(os.path.join(DATA_FOLDER, 'comp', '_'.join(['comp', group + '.csv'])), index=False)
#
#
# def split_ibes():
#     ibes = pd.read_csv(os.path.join(DATA_FOLDER, 'ibes', 'ibesPE_2ind.csv'))
#     for group in groups:
#         icm_group = icm_raw[icm_raw['PERMNO'].apply(lambda _: _[:2]) == group]
#         ibes_group = ibes[ibes['TICKER'].isin(list(icm_group['TICKER']))]
#         ibes_group.reset_index(drop=True, inplace=True)
#         ibes_group.to_csv(os.path.join(DATA_FOLDER, 'ibes', '_'.join(['ibes', group + '.csv'])), index=False)


# Explore functions
# def comp_explore():
#     comp = load_comp('')
#     for g in list(set(comp['gvkey']))):
#         comp_g = comp.loc[comp['gvkey'] == g].copy()
#         comp_g.reset_index(inplace=True, drop=True)
#         for idx in range(len(comp_g) - 1):
#             diff = comp_g.iloc[[idx + 1], :]['fyear'] - comp_g.iloc[[idx], :]['fyear']
#             if diff != 1: print(pd.concat([comp_g.iloc[[idx], :], comp_g.iloc[[idx + 1], :]])); print('\n')
