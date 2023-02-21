import pandas as pd
import numpy as np
import scipy as sp
import sys
from matplotlib import pyplot as plt

'''
Utility to identify highly variable sites within HLA sequences.
'''

if __name__ == '__main__':
    aa_file = 'data/amino_acid_ordering.txt'
    hla_file = 'data/alleles_95_aligned.txt'
    hla_variable_output = 'data/alleles_95_variable.txt'
    variable_sites_input = 'data/psuedo_sequence.txt'
    variable_sites_output = 'data/variable_sites.txt'

    aa_map = pd.read_csv(aa_file, header=None, names=['aa']).set_index('aa')
    aa_map['index'] = np.arange(0,22)

    vsites = pd.read_csv(variable_sites_input, header=None, names=['sites'])

    hla_sequences = pd.read_csv(hla_file, header=None, names=['allele','sequence'])

    hla_seq_table = hla_sequences['sequence'].apply(list).apply(pd.Series)
    variable_sites = vsites['sites'].values - 1
    hla_sequences['variable_sequence'] = hla_seq_table[variable_sites].apply(''.join, axis=1)

    hla_sequences.loc[:,['allele','variable_sequence']].to_csv(hla_variable_output, index=False, header=None)
    pd.DataFrame(variable_sites).to_csv(variable_sites_output, index=False, header=None)

'''
if __name__ == '__main__':
    aa_file = 'data/amino_acid_ordering.txt'
    hla_file = 'data/alleles_95_aligned.txt'
    hla_variable_output = 'data/alleles_95_variable.txt'
    variable_sites_output = 'data/variable_sites.txt'

    aa_map = pd.read_csv(aa_file, header=None, names=['aa']).set_index('aa')
    aa_map['index'] = np.arange(0,22)

    hla_sequences = pd.read_csv(hla_file, header=None, names=['allele','sequence'])

    hla_seq_table = hla_sequences['sequence'].apply(list).apply(pd.Series)
    hla_seq_freq = hla_seq_table.apply(pd.value_counts).apply(lambda x: x.div(x.sum())).fillna(0)

    entropy = hla_seq_freq.apply(sp.stats.entropy)
    variable_sites = np.where(entropy > 0.2)[0]
    hla_sequences['variable_sequence'] = hla_seq_table[variable_sites].apply(''.join, axis=1)

    hla_sequences.loc[:,['allele','variable_sequence']].to_csv(hla_variable_output, index=False, header=None)
    pd.DataFrame(variable_sites).to_csv(variable_sites_output, index=False, header=None)
'''