import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#loading the dataset
data = pd.read_csv('creditcard.csv')

"""
#exploring the dataset
print("Columns in dataset -  ", data.columns)
print(data.shape())
print(data.describe())


data.hist(figsize=(20,20))
plt.savefig('histogram.png')
plt.show()

fraud = data[data['Class']==1]
valid = data[data['Class']==0]

print("Number of fraud transactions- ", len(fraud))
print("Number of valid transactions- ", len(valid))
print("Ratio of fraudulent transactions- ", len(fraud) / (len(fraud) + len(valid)))
"""

corrmat = data.corr()
fig = plt.figure(figsize=(12,9))
sns.heatmap(corrmat, vmax=.8, square=True)
plt.savefig('heatmap.png')
plt.show()
