#A script that gathers results from csv and plots a series of bar charts using Seaborn

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

cwd = os.getcwd()
os.chdir("/Users/clementmanger/Desktop/Thesis/Data")

df = pd.read_csv('Resultsfinal.csv', index_col=False)
df.columns

experiments = []

experiments.append(pd.melt(df[0:2], id_vars=['Spec']))

experiments.append(pd.melt(df[2:4], id_vars=['Spec']))

experiments.append(pd.melt(df[4:7], id_vars=['Spec']))

experiments.append(pd.melt(df[7:10], id_vars=['Spec']))

experiments.append(pd.melt(df[10:13], id_vars=['Spec']))

experiments.append(pd.melt(df[13:16], id_vars=['Spec']))

experiments.append(pd.melt(df[16:19], id_vars=['Spec']))

experiments.append(pd.melt(df[19:23], id_vars=['Spec']))

experiments.append(pd.melt(df[23:25], id_vars=['Spec']))

experiments[2]

for x in experiments:

    sns.barplot(x='Spec', y='value', hue='variable', data=x, palette="Blues_d")
    plt.ylabel('Test Batch Accuracy %')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=5, mode="expand", borderaxespad=0.)
    plt.show()

os.chdir("/Users/clementmanger/Desktop/Timers")
d = {}

for x in os.listdir():
    df = pd.read_csv(x)
    df['factor'] = df['Steps']/50
    df = df.drop([0])
    df['50 Steps'] = df['Time']/df['factor']

    d.update({x[:-9]: df['50 Steps']})

timings = pd.DataFrame.from_dict(d)


experiments = []
experiment1 = []
experiment1.append(timings.columns[0])
experiment1.append(timings.columns[1])
experiments.append(experiment1)
experiment8
experiment2 = []
experiment2.append(timings.columns[3])
experiment2.append(timings.columns[1])
experiment2.append(timings.columns[4])
experiments.append(experiment2)

experiment3 = []
experiment3.append(timings.columns[7])
experiment3.append(timings.columns[1])
experiment3.append(timings.columns[10])
experiments.append(experiment3)


experiment4 = []
experiment4.append(timings.columns[5])
experiment4.append(timings.columns[1])
experiment4.append(timings.columns[8])
experiments.append(experiment4)

experiment5 = []
experiment5.append(timings.columns[6])
experiment5.append(timings.columns[1])
experiment5.append(timings.columns[9])
experiments.append(experiment5)

experiment6 = []
experiment6.append(timings.columns[1])
experiment6.append(timings.columns[15])
experiment6.append(timings.columns[14])
experiment6.append(timings.columns[-1])
experiments.append(experiment6)

experiment7 = []
experiment7.append(timings.columns[-3])
experiment7.append(timings.columns[1])
experiment7.append(timings.columns[-2])
experiments.append(experiment7)

experiment8 = []
experiment8.append(timings.columns[1])
experiment8.append(timings.columns[16])
experiments.append(experiment8)

experiment9 = []
experiment9.append(timings.columns[1])
experiment9.append(timings.columns[2])
experiment9.append(experiment9)

from scipy import stats

for x in experiments:
    df = timings[x]
    for c in df:
        print(c, stats.describe(df[c])[1:4])

for x in experiments:
    df = timings[x]
    df = pd.melt(df)
    sns.violinplot(data=df, y='variable', x='value')
    plt.xlabel('Time (Seconds) for 50 Steps')
    plt.ylabel('Model Spec')
    plt.show()
