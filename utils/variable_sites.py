import pandas as pd
import numpy as np
import scipy as sp
import sys
from matplotlib import pyplot as plt
import numpy as np


def fill_variable_values(data):
    variable = np.loadtxt('data/variable_sites.txt', dtype='int32')
    all_sites = np.arange(0,variable[-1]+1, dtype='int32')

    test_data = np.arange(0,len(variable))

    df_sites = pd.DataFrame(index=all_sites)
    df_sites.loc[:, 'values'] = 0
    df_sites.loc[variable, 'values'] = test_data

    return df_sites['values'].values
