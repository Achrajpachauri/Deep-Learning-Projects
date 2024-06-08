import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Perceptron

df = pd.read_csv(r'C:\Users\achra\OneDrive\Desktop\Generative AI\Deep learning\Deep-Learning- Neural Network\placement.csv')

#print(df)

#sns.scatterplot(df['cgpa'],df['resume_score'],hue=df['placed'])

X = df.iloc[:,0:2]
y = df.iloc[:,-1]

print(X)
print(y)


p = Perceptron()
p.fit(X,y)

print(p.coef_)