from collections import namedtuple, Counter
import numpy as np
from ..preprocessing.build_micro_dist import rh_to_str, r_to_str, t_to_str
from .knapsack_utils import make_one_hot_np
from .census_utils import get_rh_counts, get_over_18_counts, get_sex_total, get_age_eth, get_types, get_num_hhs, RACE_HIS_ENUM, Race, TYPES, TYPE_INDEX

MAX_LEVEL = 3

def encode_row(row):
    rh_counts = get_rh_counts(row)
    age_race = get_over_18_counts(row)
    age_eth = (get_age_eth(row),)
    sex_counts = get_sex_total(row)
    type_encoding = get_types(row)
    num_hh = (get_num_hhs(row),)
    # print([(t, te) for te, t in zip(type_encoding, TYPES) if te > 0])
    #TODO add the calls for sex, age bucket and change return below to be Encoding0
    return Encoding0(*(rh_counts + age_race + age_eth + sex_counts + type_encoding + num_hh))

def encode_hh_dist(dist):
    new_dist = Counter()
    for hh, prob in dist.items():
        # Race X eth pairs
        rh_counts = hh.rh_counts
        age_race = tuple(hh.n_over_18 * make_one_hot_np(hh.holder.race.value-1, len(Race)))
        age_eth = (hh.holder.eth * hh.n_over_18,)
        #check sex counts equation?
        sex_counts = (hh.holder.sex,)
        type_encoding = make_one_hot_np(TYPE_INDEX[hh.race_type], len(TYPE_INDEX))
        if hh.holder.eth == 1:
            type_encoding += make_one_hot_np(TYPE_INDEX[hh.eth_type], len(TYPE_INDEX))
        type_encoding = tuple(type_encoding)
        num_hh = (1,)
        new_dist[Encoding0(*(rh_counts + age_race + age_eth + sex_counts + type_encoding + num_hh))] += prob
    return new_dist


#TODO: Need to create Encoding0 that inclues sex, and age bucket? 

class Encoding0(namedtuple('Encoding0',
        [rh_to_str(rh) for rh in RACE_HIS_ENUM] +\
        ['n_18_' + r_to_str(r) for r in Race] +\
        ['n_18_HISP'] +\
        ['sex'] +\
        [t_to_str(t) for t in TYPES] +\
        ['num_hh'])):

    def get_type(self):
        # Type is defined as (Race, is_family, size, ethnicity)
        type_range = [getattr(self, t_to_str(t)) for t in TYPES]
        the_type = None
        for i, t in zip(type_range, TYPES):
            if i == 1:
                the_type = t
                break
        assert the_type is not None
        # Ethnicity as last field
        if sum(type_range) == 2:
            the_type += (1,)
        else:
            the_type += (0,)
        return the_type

    def reduce(self, level, use_age):
        if level == 0:
            # rh_counts = get_rh_counts(row)
            # age_race = get_over_18_counts(row)
            # age_eth = (get_age_eth(row),)
            # type_encoding = get_types(row)
            # num_hh = (get_num_hhs(row),)
            # schema: (r X h for all r, h; n18+ (?); num HH)
            rh_counts = tuple(getattr(self, rh_to_str(rh)) for rh in RACE_HIS_ENUM)
            age_race = tuple(getattr(self, 'n_18_' + r_to_str(r)) for r in Race)
            age_eth = tuple(getattr(self, 'n_18_HISP'))
            type_encoding = tuple(getattr(self, t_to_str(t)) for t in TYPES)
            # num HH
            hh = (self.num_hh,)
            tup = rh_counts + age_race + age_eth + type_encoding + hh
            return Encoding1(*tup)
        else:
            raise ValueError('Level %d undefined' % level)

    def to_sol(self):
        r_counts = self.get_r_counts()
        eth = (self.get_eth_count(),)
        age = (self.get_n18(),)
        
        return r_counts + eth + age

    def get_r_counts(self):
        eth_0 = np.array([getattr(self, rh_to_str((r, 0))) for r in Race], dtype=int)
        eth_1 = np.array([getattr(self, rh_to_str((r, 1))) for r in Race], dtype=int)
        r_counts = tuple(eth_0 + eth_1)
        return r_counts

    def get_eth_count(self):
        eth_1 = np.array([getattr(self, rh_to_str((r, 1))) for r in Race], dtype=int)
        return sum(eth_1)

    def get_n18(self):
        return sum(getattr(self, 'n_18_' + r_to_str(r)) for r in Race)
    
    def get_sex(self):
        return sum(getattr(self, 'sex'))

    def __str__(self):
        d = self._asdict()
        return str({k: v for k, v in d.items() if v > 0})

class Encoding1(namedtuple('Encoding1',
        [rh_to_str(rh) for rh in RACE_HIS_ENUM] +\
        ['n_18_' + r_to_str(r) for r in Race] +\
        ['n_18_HISP'] +\
        [t_to_str(t) for t in TYPES] +\
        ['num_hh'])):

    def get_type(self):
        # Type is defined as (Race, is_family, size, ethnicity)
        type_range = [getattr(self, t_to_str(t)) for t in TYPES]
        the_type = None
        for i, t in zip(type_range, TYPES):
            if i == 1:
                the_type = t
                break
        assert the_type is not None
        # Ethnicity as last field
        if sum(type_range) == 2:
            the_type += (1,)
        else:
            the_type += (0,)
        return the_type

    def reduce(self, level, use_age):
        if level == 2:
            # schema: (r X h for all r, h; n18+ (?); num HH)
            rh_counts = tuple(getattr(self, rh_to_str(rh)) for rh in RACE_HIS_ENUM)
            if use_age:
                age = (sum(getattr(self, 'n_18_' + r_to_str(r)) for r in Race),)
            else:
                age = ()
            # num HH
            hh = (self.num_hh,)
            tup = rh_counts + age + hh
            if use_age:
                return Encoding2a(*tup)
            else:
                return Encoding2b(*tup)
        elif level == 3:
            # schema: (r for all r; h; n18+ (?))
            r_counts = self.get_r_counts()
            eth = (self.get_eth_count(),)
            if use_age:
                age = (self.get_n18(),)
            else:
                age = ()
            tup = r_counts + eth + age
            if use_age:
                return Encoding3a(*tup)
            else:
                return Encoding3b(*tup)
        else:
            raise ValueError('Level %d undefined' % level)

    def to_sol(self):
        r_counts = self.get_r_counts()
        eth = (self.get_eth_count(),)
        age = (self.get_n18(),)
        return r_counts + eth + age

    def get_r_counts(self):
        eth_0 = np.array([getattr(self, rh_to_str((r, 0))) for r in Race], dtype=int)
        eth_1 = np.array([getattr(self, rh_to_str((r, 1))) for r in Race], dtype=int)
        r_counts = tuple(eth_0 + eth_1)
        return r_counts

    def get_eth_count(self):
        eth_1 = np.array([getattr(self, rh_to_str((r, 1))) for r in Race], dtype=int)
        return sum(eth_1)

    def get_n18(self):
        return sum(getattr(self, 'n_18_' + r_to_str(r)) for r in Race)

    def __str__(self):
        d = self._asdict()
        return str({k: v for k, v in d.items() if v > 0})

class Encoding2a(namedtuple('Encoding2a',
        [rh_to_str(rh) for rh in RACE_HIS_ENUM] +\
        ['n_18'] +\
        ['num_hh'])):

    def get_r_counts(self):
        eth_0 = np.array([getattr(self, rh_to_str((r, 0))) for r in Race], dtype=int)
        eth_1 = np.array([getattr(self, rh_to_str((r, 1))) for r in Race], dtype=int)
        r_counts = tuple(eth_0 + eth_1)
        return r_counts

    def get_eth_count(self):
        eth_1 = np.array([getattr(self, rh_to_str((r, 1))) for r in Race], dtype=int)
        return sum(eth_1)

    def to_sol(self):
        r_counts = self.get_r_counts()
        eth = (self.get_eth_count(),)
        age = (self.n_18,)
        return r_counts + eth + age

    def __str__(self):
        d = self._asdict()
        return str({k: v for k, v in d.items() if v > 0})

class Encoding2b(namedtuple('Encoding2b',
        [rh_to_str(rh) for rh in RACE_HIS_ENUM] +\
        ['num_hh'])):

    def get_r_counts(self):
        eth_0 = np.array([getattr(self, rh_to_str((r, 0))) for r in Race], dtype=int)
        eth_1 = np.array([getattr(self, rh_to_str((r, 1))) for r in Race], dtype=int)
        r_counts = tuple(eth_0 + eth_1)
        return r_counts

    def get_eth_count(self):
        eth_1 = np.array([getattr(self, rh_to_str((r, 1))) for r in Race], dtype=int)
        return sum(eth_1)

    def to_sol(self):
        r_counts = self.get_r_counts()
        eth = (self.get_eth_count(),)
        return r_counts + eth

    def __str__(self):
        d = self._asdict()
        return str({k: v for k, v in d.items() if v > 0})

class Encoding3a(namedtuple('Encoding3a',
        [r_to_str(r) for r in Race] +\
        ['num_hisp'] +\
        ['n_18'])):

    def to_sol(self):
        return self

    def __str__(self):
        d = self._asdict()
        return str({k: v for k, v in d.items() if v > 0})

class Encoding3b(namedtuple('Encoding3b',
        [r_to_str(r) for r in Race] +\
        ['num_hisp'])):

    def to_sol(self):
        return self

    def __str__(self):
        d = self._asdict()
        return str({k: v for k, v in d.items() if v > 0})

