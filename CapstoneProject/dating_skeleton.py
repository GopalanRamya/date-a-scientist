import sys
import time
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn import preprocessing
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import precision_recall_curve
from sklearn.utils.fixes import signature
from sklearn.metrics import confusion_matrix

#Create your df here:
df = pd.read_csv('profiles.csv')
df.dropna(inplace=True)

#### Data augmentation ####

### Augmentation 1: Augment body_type column
bodyType = {'a little extra':1, 'average':2, 'thin':3, 'athletic':4, 'fit':5, 'skinny':6, 'curvy': 7,
 'full figured':8, 'jacked':9, 'rather not say':10, 'used up':11, 'overweight':12}
df.body_type.fillna('rather not say', inplace=True);
df.body_type = [bodyType[item] for item in df.body_type];
statusMap = {'single':1, 'married': 2, 'available': 3, 'seeing someone': 4, 'unknown': 5}
df.status = df.status.map(statusMap)

### Augmentation 2: Augment smokes column
df.smokes.fillna('rather not say', inplace=True);
smokesMap = {'sometimes':1, 'no':2, 'when drinking':3, 'yes':4, 'trying to quit':5, 'rather not say':6}
df.smokes = df.smokes.map(smokesMap)

### Augmentation 3: Augment drugs column
df.drugs.fillna('rather not say', inplace=True);
drugsMap = {'never':1, 'sometimes':2, 'often':3, 'rather not say':4}
df.drugs = df.drugs.map(drugsMap)

### Augmentation 4: Augment orientation column
df.orientation.fillna('rather not say', inplace=True);
orientationMap = {'straight':1, 'bisexual':2, 'gay':3}
df.orientation = df.orientation.map(orientationMap)

### Augmentation 5: Create column based on frequency count of the word "love" in essay0
df.essay0.fillna('', inplace=True);
loveCnts = df.essay0.apply(lambda x: x.count("cool"))

#### Construct Feature and label data ####
dfFeatureVector = df[['orientation', 'height']];
#dfFeatureVector['love'] = loveCnts
mean = (dfFeatureVector.mean())
variance = (dfFeatureVector.var()) ** 0.5

dfFeatureVectorScaled = preprocessing.scale(dfFeatureVector)
dfLabels = df['sex']
labelMap = {'m':0, 'f':1}
dfLabels = dfLabels.map(labelMap)


#### Data Plots ####

### Plot 1: Status, age pie chart
N = 101 # data points
xBins = [15, 35, 55, 75]; # Max and min of df.age used to determine hist boundaries
yBins = [1, 2, 3, 4, 5];
hist, xBins, yBins = np.histogram2d(df.age, df.status, bins=(xBins, yBins))
labels=['Age15-35,single', '', '', '', 'Age35-55,single', '', '', '', 'Age55-75,single', '', '', '']
labelsLegend=['Age15-35,single', 'Age15-35,married', 'Age15-35,available', 'Age15-35,seeing someone', 'Age35-55,single', 'Age35-55,married', 'Age35-55,available', 'Age35-55,seeing someone', 'Age55-75,single', 'Age55-75,married', 'Age55-75,available', 'Age55-75,seeing someone']
fig, ax = plt.subplots()
size = 0.3
cmap = plt.get_cmap("tab20c")
inner_colors = cmap(np.array(range(hist.shape[0] * hist.shape[1])))
ax.pie(hist.flatten(), radius=1.0, labels=labels, colors=inner_colors, wedgeprops=dict(width=size, edgecolor='w'))
ax.set(aspect="equal", title='Age & status pie chart')
ax.legend(labelsLegend, bbox_to_anchor=(1, 1))
plt.show()


### Plot 2: Scatter plot of height v/s income of female and male users
dfMale  = df.loc[(df['sex'].isin(['m']))]
dfMale2 = dfMale.loc[(df['income'] != -1)]
N = 100
dfMaleHeight  = dfMale2['height'].iloc[1:N]
dfMaleIncome = dfMale2['income'].iloc[1:N]
dfFemale = df.loc[(df['sex'].isin(['f']))]
dfFemale2 = dfFemale.loc[(df['income'] != -1)]
dfFemaleHeight = dfFemale2['height'].iloc[1:N]
dfFemaleIncome = dfFemale2['income'].iloc[1:N]
plt.scatter(dfFemaleHeight, dfFemaleIncome, color='tab:pink',label='female')
plt.scatter(dfMaleHeight, dfMaleIncome, color='b', alpha=0.75, label='male')
plt.legend()
plt.xlabel('Height')
plt.ylabel('Income')
plt.title('Height and income stats of 100 male and female users')
plt.show()


#### Obtain train and test data ####
[x_train, x_test, y_train, y_test] = train_test_split(dfFeatureVectorScaled, dfLabels, test_size=0.2)


#### Build classifier ####

### Classifier 1: KNN

accuracy = []
times    = []
x = list(range(1,1000))
for k in range(1,1000):
  classifier = KNeighborsClassifier(n_neighbors=k)
  classifier.fit(x_train, y_train)
  t0 = time.time()
  result = classifier.score(x_test, y_test)
  accuracy.append(result)
  times.append((time.time() - t0)/len(x_test))
plt.figure()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.plot(x,accuracy)
plt.show
plt.figure()
plt.xlabel('Number of neighbors')
plt.ylabel('time taken to score test set')
plt.plot(x,times)
plt.show()


### Classifier 2: SVM
g = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0]
result = []
times = []
for i in g:
  classifier = svm.SVC(kernel='linear',gamma=i)
  classifier.fit(x_train, y_train)
  t0 = time.time()
  result.append(classifier.score(x_test, y_test))
  times.append((time.time() - t0)/len(x_test))
plt.figure()
plt.xlabel('gamma')
plt.ylabel('Accuracy')
plt.plot(g, result)
plt.show()
plt.figure()
plt.xlabel('gamma')
plt.ylabel('time taken to score test set')
plt.plot(g, times)
plt.show()

y_score = classifier.predict(x_test)
print('SVM: confusion matrix for male female classification')
print(confusion_matrix(y_test, y_score))

dfAge = df['age']
dfICnts =  df['income'] #df.essay0.apply(lambda x: x.count(" i "))
x_train, x_test, y_train, y_test = train_test_split(dfICnts.values.reshape(-1,1), dfAge.values.reshape(-1,1), test_size=0.2)
model = LinearRegression()
model.fit(x_train, y_train)
times = []
t0 = time.time()
print('Linear Regression score for age determination')
print(model.score(x_test, y_test))
print('Linear Regression: time taken')
times.append((time.time() - t0)/len(x_test))
print(times)
print('Linear Regression: coefficient and intercept values')
print(model.coef_)
print(model.intercept_)

### Regression 2: KNN
n_neighbors = range(1,700)
scores = []
times = []
for k in n_neighbors:
  model = KNeighborsRegressor(k)
  model.fit(x_train, y_train)
  t0 = time.time()
  scores.append(model.score(x_test, y_test))
  times.append((time.time() - t0)/len(x_test)) 
plt.figure()
plt.xlabel('neighbors')
plt.ylabel('Accuracy')
plt.title('KNN Regression')
plt.plot(n_neighbors, scores)
plt.show()
plt.figure()
plt.xlabel('neighbors')
plt.ylabel('Time taken')
plt.title('KNN Regression')
plt.plot(n_neighbors, times)
plt.show()
