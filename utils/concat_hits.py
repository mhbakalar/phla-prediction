import pandas as pd

files = ['hits_95_8.txt', 'hits_95_9.txt', 'hits_95_10.txt', 'hits_95_11.txt']
#files = ['decoys_8.txt', 'decoys_9.txt', 'decoys_10.txt', 'decoys_11.txt']

dfs = []
for file in files:
    df = pd.read_csv('./data/'+file, header=None)
    dfs.append(df)

df = pd.concat(dfs)
df = df.drop(df[df[0] == 'seq'].index)
df = df.rename(columns={0:'allele', '1':'len', '2':'seq'})
df.to_csv('hits_95.txt', index=False)

pwd
df = pd.read_csv('data/hits_16.txt', sep=' ')
df
