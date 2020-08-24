# This module processes the data from WRDS and constructs the appropriate dataframe for multiples valuation
# dsf folder in the Data folder (load using the load_dsf function) contains data from 1990 - 2020
# 'E:\Git\dsf.csv' contains data from 00-17, without unnecessary columns to minimize load efficiency
# 'E:\Git\multiples' contains data from 00-17, splitted into groups, with LTM_eps & PE columns
# 'E:\Git\multiples_pickle' same data, but stored in pickle file format
# 'E:\Git\temp' Multiples 3663 First Experiment
# 'E:\Git\data\temp_yanlong' transfers to organize\PE_original
# test passed; no need to re-split

import pandas as pd
import numpy as np
import os
from tqdm.notebook import tqdm
from utils.backtest_tools import load_comp
from portfolio.val_data import ValError


# groups = [str(i) for i in range(10,94)]
def multiple_append(groups, directory):
    # the permno group loop
    # the aim of splitting: indexing is expensive for very large dataset
    # load outside the loop to increase speed
    # FOLDER = r'E:\Git\new_split'

    for group in tqdm(groups):
        comp_group = load_comp(group).set_index(['gvkey'], inplace=False)
        dsf_group = pd.read_csv(os.path.join(directory, '_'.join(['dsf', group]) + '.csv'))
        # deal with unexpected datatypes and staff
        dsf_group['PERMNO'] = dsf_group['PERMNO'].astype(str)
        # del dsf_group['Unnamed: 0']
        dsf_group['date'] = pd.to_datetime(dsf_group['date'], format='%Y-%m-%d')
        dsf_group['SICCD'] = dsf_group['SICCD'].astype(str).apply(lambda _: _.zfill(4))

        # deal with the price prediction (negative value) problem
        dsf_group['PRC'] = dsf_group['PRC'].apply(abs)
        # Preparation
        # dsf_group.set_index(['PERMNO'],inplace=True)
        permno_array = dsf_group['PERMNO'].unique()
        LTM_bvps = []

        # the permno & date loop
        # here used the hist_ function to get the BVPS instead of EPS
        for p in tqdm(permno_array):
            filt = dsf_group['PERMNO'] == p
            for buy_date in dsf_group.loc[filt]['date']:
                try:
                    comp_func, data_group = [hist_, [comp_group]]
                    bvps_0_, bvps_1_, bvps_2_, bvps_3_, sic_, end_date = comp_func(buy_date, data_group, p)
                    if not (np.isnan(bvps_0_) or np.isnan(bvps_1_)):
                        pct = int((buy_date - end_date).days) / 365
                        LTM_bvps.append(bvps_0_ * (1 - pct) + bvps_1_ * pct)
                    else:
                        LTM_bvps.append(np.nan)
                except (KeyError, ValError):
                    LTM_bvps.append(np.nan)
                    pass
        dsf_group['LTM_bvps'] = LTM_bvps
        dsf_group['P/B'] = dsf_group['PRC'] / dsf_group['LTM_bvps']
        dsf_group.to_csv(os.path.join(directory, '_'.join(['dsf', group + '.csv'])), index=False)


# obtain the SICCD code lsit

list_file = open(r'E:\Git\data\temp_PB\your_file.txt')
content = list_file.read()
siccd_list = content.split(' ')


def multiple_df(siccd_list,directory):
    # read_csv place out of the loop to reduce time complexity, or does it really save time?
    # prepare them for further usage
    # to run test on two of them
    # for testing: siccd_list = ['0131', '2328']
    # Things to edit: FOLDER & To_csv on the last line, list_file in the previous cell

    siccd_final = []
    date_final = []
    median_final = []
    # directory = r'E:\Git\multiples_pickle'
    for siccd in tqdm(siccd_list):
        industry_df = pd.DataFrame()
        groups = [str(i) for i in range(10, 94)]
        for group in groups:
            dsf_group = pd.read_pickle(os.path.join(directory, '_'.join(['dsf', group]) + '.pkl'))
            dsf_group['date'] = pd.to_datetime(dsf_group['date'], format='%Y-%m-%d')
            dsf_group['SICCD'] = dsf_group['SICCD'].astype(str)
            # choose the siccd
            siccd_filt = dsf_group['SICCD'] == siccd
            dsf_group = dsf_group.loc[siccd_filt]
            industry_df = industry_df.append(dsf_group)

        date_array = industry_df['date'].unique()
        median_multiple = []
        for date in date_array:
            date_filt = industry_df['date'] == date
            median_multiple.append(np.median(industry_df[date_filt]['P/B'].values))

        siccd_col = [siccd for _ in range(len(median_multiple))]
        siccd_final += siccd_col
        date_final += list(date_array)
        median_final += median_multiple
    PB_df = pd.DataFrame({'SICCD': siccd_final, 'date': date_final, 'median_PB': median_final})
    PB_df.to_csv(os.path.join(directory, 'PB.csv'), index=False)


def multiple_dropna(groups, directory):
    # the to_csv index=False avoid Pandas creating an index in a saved csv
    # dir = r'E:\Git\multiples'
    for group in tqdm(groups):
        dsf_group = pd.read_csv(os.path.join(directory, '_'.join(['dsf', group]) + '.csv'))
        dsf_group.dropna(axis='index', how='any', inplace=True)
        dsf_group.to_csv(os.path.join(directory, '_'.join(['dsf', group + '.csv'])), index=False)


def multiple_topkl(groups, file_dir, target_dir):
    # file_dir = r'E:\Git\multiples'
    # target_dir = r'E:\Git\multiples_pickle'
    for group in tqdm(groups):
        dsf_group = pd.read_csv(os.path.join(file_dir, '_'.join(['dsf', group]) + '.csv'))
        dsf_group.to_pickle(os.path.join(target_dir, '_'.join(['dsf', group + '.pkl'])))