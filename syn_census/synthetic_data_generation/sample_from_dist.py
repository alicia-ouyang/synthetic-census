import os
import pickle
import re
import numpy as np
from collections import Counter
import pandas as pd
from ..utils.knapsack_utils import normalize
from ..utils.census_utils import Race
from ..utils.config2 import ParserBuilder
from ..preprocessing.build_micro_dist import read_microdata
from ..preprocessing.build_block_df import make_identifier_non_unique

parser_builder = ParserBuilder({
    'micro_file': True,
    'block_clean_file': True,
    'synthetic_output_dir': True,
    'synthetic_data': True,
    'task_name': False,
    })

POP_COLS = {
        'H7X002',
        'H7X003',
        'H7X004',
        'H7X005',
        'H7X006',
        'H7X007',
        'H7X008',
        }
SHORT_RN = {
        'H7X001': 'TOTAL',
        'H7X002': 'W',
        'H7X003': 'B',
        'H7X004': 'AI_AN',
        'H7X005': 'AS',
        'H7X006': 'H_PI',
        'H7X007': 'OTH',
        'H7X008': 'TWO_OR_MORE',
        'H76002': 'MALE', 
        'H76026': 'FEMALE',
        'H8A003': 'BLOCK_18_PLUS',
        }
GENDER_COLS = {}
RACE_MAP = {
        Race.WHITE: 'W',
        Race.BLACK: 'B',
        Race.AM_IND_ALASKAN: 'AI_AN',
        Race.ASIAN: 'AS',
        Race.HAWAIIAN_PI: 'H_PI',
        Race.OTHER: 'OTH',
        Race.TWO_PLUS: 'TWO_OR_MORE',
        }

DEMO_COLS = [SHORT_RN[k] for k in sorted(SHORT_RN.keys())[1:-1]] + ['NUM_HISP', '18_PLUS', 'GENDER']

OUTPUT_COLS = [
        'YEAR',
        'STATE',
        'STATEA',
        'COUNTY',
        'COUNTYA',
        'COUSUBA',
        'TRACTA',
        'BLKGRPA',
        'BLOCKA',
        'NAME',
        'BLOCK_TOTAL',
        'H8A003',
        'H7X001',
        'H7X002',
        'H7X003',
        'H7X004',
        'H7X005',
        'H7X006',
        'H7X007',
        'H7X008',
        'GENDER',
        'NUM_HISP',
        '18_PLUS',
        'HH_NUM',
        'ACCURACY',
        'AGE_ACCURACY',
        ]

RECORD_LENGTH = len(Race) + 3

CARRYOVER = set(OUTPUT_COLS) - POP_COLS

def process_dist(dist):
    new_dist = Counter()
    for tup, prob in dist.items():
        new_dist[tup.race_counts + (tup.eth_count, tup.n_over_18)] += prob
    return normalize(new_dist)

def load_sample_and_accs(task_name, dist, out_dir):
    sample = {}
    accs = {}
    errors = []
    for fname in sorted(os.listdir(out_dir)):
        if re.match(task_name + '[0-9]+_[0-9]+.pkl', fname):
            print('Reading from', out_dir+fname)
            with open(out_dir + fname, 'rb') as f:
                result_list = pickle.load(f)
                for results in result_list:
                    breakdown = results['sol']
                    print("breakdown length:", len(breakdown[0]), "record length", RECORD_LENGTH)
                    # Add age if missing
                    #checking if indicies are less than 1 for age?
                    if breakdown[0][RECORD_LENGTH-2] ==  - 1:
                        breakdown = add_age(breakdown, dist)
                        print("AGE was added new breakdown", breakdown)
                    #check if the sex index is -1
                    if breakdown[0][RECORD_LENGTH-1] ==  - 1:
                        breakdown = add_sex(breakdown, dist)
                    sample[results['id']] = breakdown
                    accs[results['id']] = (results['level'], results['age'])
    return sample, accs, errors

def add_age(hh_list, dist):
    out_list = []
    for hh in hh_list:
        print("Household vector for adding age", hh)
        hh_sex = hh[-1]
        print("Double check sex value", hh_sex)
        hh = hh[:-2]
        eligible = [full_hh for full_hh in dist if hh == full_hh[:RECORD_LENGTH-2]]
        probs = np.array([dist[full_hh] for full_hh in eligible], dtype='float')
        ps = probs / np.sum(probs)
        out_list.append(eligible[np.random.choice(range(len(eligible)), p=ps)])
        print("CHECKING ADD AGE OUTPUTS", out_list, (tuple(sorted(out_list[0]))+(hh_sex,),))
    return (tuple(sorted(out_list[0]))+(hh_sex,),)

#Writing add_sex TODO Needs to be pulling from ipums person instead of household, might need to look at read_microdata
def add_sex(hh_list, dist):
    out_list = []
    for hh in hh_list:
        print("Household vector for adding sex", hh)
        hh = hh[:-1]
        eligible = [full_hh for full_hh in dist if hh == full_hh[:RECORD_LENGTH-1]]
        print('ADD SEX DEBUGGING, HOW MANY ELIGIBLE', len(eligible))
        probs = np.array([dist[full_hh] for full_hh in eligible], dtype='float')
        ps = probs / np.sum(probs)
        out_list.append(eligible[np.random.choice(range(len(eligible)), p=ps)])
    return tuple(sorted(out_list))


def aggregate_shards(
        micro_file: str,
        block_clean_file: str,
        synthetic_output_dir: str,
        task_name: str,
        ):
    df = pd.read_csv(block_clean_file)
    print(df.head())
    dist = read_microdata(micro_file)
    dist = process_dist(dist)
    sample, accs, errors = load_sample_and_accs(task_name, dist, synthetic_output_dir)
    df_dict = {col: [] for col in OUTPUT_COLS}
    for ind, row in df.iterrows():
        if row['identifier'] in errors:
            print('Error index', ind)
            continue
        try:
            breakdown = sample[row['identifier']]
        except Exception as e:
            print(e)
            1/0
            continue
        r_list = [row[x] for x in df.columns if x in CARRYOVER]
        try:
            for i,b in enumerate(breakdown):
                cur_row = r_list + [sum(b[:-2])] + list(b) + [i] + list(accs[row['identifier']])
                for i, col in enumerate(OUTPUT_COLS):
                    df_dict[col].append(cur_row[i])
        except Exception as e:
            print('Error:', ind)
            print(breakdown)
            print(e)
            continue
    out_df = pd.DataFrame.from_dict(df_dict)[OUTPUT_COLS]
    out_df.rename(columns=SHORT_RN, inplace=True)
    make_identifier_non_unique(out_df)
    print(out_df.head())
    assert len(out_df['identifier'].unique()) == len(df['identifier'].unique()), 'Not all blocks found'
    return out_df
