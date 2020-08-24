import pandas as pd


def clean_sp500(sp500):
    sp500['Date'] = pd.to_datetime(sp500['Date'])
    sp500['Return'] = sp500['Adj Close'].pct_change()
    sp500.loc[0, 'Return'] = 0
    sp500['Cumprod'] = sp500['Return'].add(1).cumprod()

    return sp500


def clean_ccm(ccm):
    ccm['permno'] = ccm['permno'].astype(int).astype(str)
    ccm['gvkey'] = ccm['gvkey'].astype(str)
    ccm['linkdt'] = pd.to_datetime(ccm['linkdt'])
    ccm['linkenddt'] = pd.to_datetime(ccm['linkenddt'])
    isnull = ccm['linkenddt'].isnull()
    ccm.loc[isnull, 'linkenddt'] = pd.Timestamp(year=2047, month=7, day=1)

    return ccm


def clean_icm(icm):
    icm.drop_duplicates(inplace=True)
    icm['PERMNO'] = icm['PERMNO'].astype(int).astype(str)
    icm['TICKER'] = icm['TICKER'].astype(str)
    icm['NCUSIP'] = icm['NCUSIP'].astype(str)
    icm['sdate'] = pd.to_datetime(icm['sdate'], format='%d%b%Y')
    icm['edate'] = pd.to_datetime(icm['edate'], format='%d%b%Y')
    icm = icm.loc[icm['SCORE'] == 1, :]
    icm.reset_index(drop=True, inplace=True)

    return icm


def clean_link(link):
    link['PERMNO'] = link['PERMNO'].astype(int).astype(str)
    link['namedt'] = pd.to_datetime(link['namedt'])
    link['nameenddt'] = pd.to_datetime(link['nameenddt'])

    return link


def clean_ff(ff):
    ff.index = pd.to_datetime(ff.index)
    ff['RF'] = ff['RF'] / 100

    return ff
