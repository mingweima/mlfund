from global_tools import clean_sp500, clean_ccm, clean_icm, clean_link, clean_ff
import pandas as pd
import os

#TODO: edit the SOURCE_FOLDER
user = os.getcwd()
# assert user in ['/Users/mmw/Documents/GitHub/ml_val', 'E:\\Git\\ml_val', '/Users/mingyu/Desktop/ml_val']

if user == '/Users/mmw/Documents/GitHub/ml_val':
    DATA_FOLDER = '/Volumes/T5/data_all'
    SOURCE_FOLDER = '/Users/mmw/Documents/GitHub/ml_val'
elif user == 'E:\\Git\\ml_val':
    DATA_FOLDER = 'E:\\Git\\data'
    SOURCE_FOLDER = 'E:\\Git\\ml_val'
elif user == '/home/mma3/ml_val':
    DATA_FOLDER = '/home/mma3/mvdata'
    SOURCE_FOLDER = '/home/mma3/ml_val'
else:
    DATA_FOLDER = '/Users/mingyu/Desktop/data'
    SOURCE_FOLDER = '/Users/mingyu/Desktop/ml_val'

LOOKUP_FOLDER = os.path.join(SOURCE_FOLDER, '../../lookup')
CONFIG_FOLDER = os.path.join(SOURCE_FOLDER, 'config')


# SP500
sp500 = pd.read_csv(os.path.join(LOOKUP_FOLDER, 'sp500.csv'))
sp500 = clean_sp500(sp500)
ff = pd.read_csv(os.path.join(LOOKUP_FOLDER, 'FF.csv'), index_col=0)
ff = clean_ff(ff)

# CCM Link -- until end of 2018
ccm = pd.read_pickle(os.path.join(LOOKUP_FOLDER, 'ccm.pkl'))
ccm_raw = pd.read_pickle(os.path.join(LOOKUP_FOLDER, 'ccm_raw.pkl'))
ccm, ccm_raw = clean_ccm(ccm), clean_ccm(ccm_raw)
groups = sorted(list(set([str(_)[:2] for _ in ccm['permno']])))
SIC1 = pd.read_pickle(os.path.join(LOOKUP_FOLDER, 'sic1.pkl'))
SIC2 = pd.read_pickle(os.path.join(LOOKUP_FOLDER, 'sic2.pkl'))
GIC1 = pd.read_pickle(os.path.join(LOOKUP_FOLDER, 'gic1.pkl'))
GIC2 = pd.read_pickle(os.path.join(LOOKUP_FOLDER, 'gic2.pkl'))
PERMNO = sorted(list(set(ccm_raw['permno'])))


# ICM Link -- until end of 2018
icm_raw = pd.read_pickle(os.path.join(LOOKUP_FOLDER, 'icm_raw.pkl'))
icm_raw = clean_icm(icm_raw)

# DCX Link
link_now = pd.read_csv(os.path.join(LOOKUP_FOLDER, 'link_now.txt'), sep='\t', encoding='latin1', dtype='str')
link_raw = pd.read_csv(os.path.join(LOOKUP_FOLDER, 'link_raw.txt'), sep='\t', encoding='latin1', dtype='str')
link_now, link_raw = clean_link(link_now), clean_link(link_raw)
