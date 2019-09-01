import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

df = pd.read_csv("data.csv")
df = df.drop("name", axis=1)

array = df.to_numpy()


pred = KMeans(n_clusters=4).fit_predict(array)
print(pred)