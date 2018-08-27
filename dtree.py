from sklearn.externals import joblib
import pandas as pd 
import matplotlib.pyplot as plt 
import math
from curve_classify import pre_process
import numpy as np 
import os
from sklearn.ensemble import RandomForestClassifier

cluster_num = 11

data = pd.read_csv('features_withid.csv')
features, filename_list = [], []
for i in range(len(data[data.columns[1]])):
	feature = np.array(data.iloc[i])
	filename_list.append(feature[0])
	features.append(feature[1:])

clf = joblib.load('trained_model.pkl')
pred = clf.predict(features)
'''
print(features)
print(filename_list)
print(pred)
data = pd.DataFrame()
data['filename'] = filename_list
data['class'] = pred
data.to_csv('classification.csv')
namelist = []
for i in range(11):
	name = []
	for j in range(len(pred)):
		if pred[j] == i:
			name.append(filename_list[j])
	namelist.append(name)
print(namelist)
'''
model = RandomForestClassifier(max_depth = 5)
model.fit(features, pred)
pred1 = model.predict(features)

index = []
cnt = 0
for i in range(len(pred)):
	if pred[i] != pred1[i]:
		index.append(i)
		cnt += 1
print(cnt)

index = []
for i in range(100):
	index.append(int(1000 * np.random.rand()))
features, filename_list = [], []
for i in index:
	feature = np.array(data.iloc[i])
	filename_list.append(feature[0])
	features.append(feature[1:])

pred = model.predict(features)

def visual(filename_list, pred, index = 1, Length = 3000):
	classindex = []
	for c in range(cluster_num):
		temp = []
		for i in range(len(pred)):
			if pred[i] == c:
				temp.append(i)
		classindex.append(temp)

	ind = 0
	for i in range(len(classindex)):
		plt.figure()
		plt.suptitle('Class: "{}"'.format(i))
		L = classindex[i]
		N = math.sqrt(len(L)) + 1
		for i in range(len(L)):
			data = pd.read_csv(filename_list[L[i]])
			if len(data) > Length:
				data = data[:Length]
			value = data[data.columns[index]]
			value = np.array(value)
			plt.subplot(N, N, i + 1)
			plt.xticks([])
			plt.yticks([])
			plt.plot(value)
			plt.plot(pre_process(value, 200, _normalize = False))
		plt.savefig('Figure_for_Class{}'.format(ind), dpi = 600)
		ind += 1

visual(filename_list, pred)