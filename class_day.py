from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
import numpy as np 
import pandas as pd 
import math
import os
import matplotlib.pyplot as plt 
from classification import get_average
from classify import visual, visual_withseason, visual_justdiff
from sklearn.cluster import Birch, DBSCAN, AffinityPropagation, KMeans
from HoltWinters import holtWinters_forclass

feature_path = 'features_season.csv'
cluster_num = 2
feature_name = ['skewness', 'kurtosis', 'variance', 'mean', 'cnt_above_mean', 'lgst_above_mean', 'median', 'abs_change', 'cnt_turn', 'cnt_anomaly', 'amplitude']
weights = [0,0,0,0,0,0,0,0,0,5,0]


def _cal(feature, weights):
	res = []
	for i in range(len(weights)):
		if weights[i] != 0:
			res.append(feature[i] * weights[i])
	return np.array(res)

def train(feature_path, cluster_num, weights):
	data = pd.read_csv(feature_path)
	features, filename_list = [], []
	if len(weights) != (len(data.iloc[1]) - 1):
		print("Length of weights not match!!")
		return
	for i in range(len(data)):
		feature = np.array(data.iloc[i])
		features.append(_cal(feature[1:], weights))
		filename_list.append(feature[0])
	#plt.plot(features, np.full(len(features), 1), 'bo', markersize = 1)
	#plt.show()
	model = Birch(n_clusters = cluster_num)
	model = KMeans(n_clusters = cluster_num)
	pred = model.fit_predict(features)
	print(pred)
	joblib.dump(model, 'model_season.pkl')
	return max(pred)

def pred(feature_path, cluster_num, n_samples, weights):
	data = pd.read_csv(feature_path)
	features, filename_list = [], []
	if len(weights) != (len(data.iloc[1]) - 1):
		print("Length of weigths not match!!")
		return
	index = np.random.randint(0, 940, n_samples)
	for i in index:
		feature = np.array(data.iloc[i])
		features.append(_cal(feature[1:], weights))
		filename_list.append(feature[0])
	model = joblib.load('model_season.pkl')
	pred = model.predict(features)
	visual_withseason(filename_list, pred, cluster_num, Length = 3000, index = 1)
	'''
	color = ['red', 'blue', 'green', 'yellow']
	plt.figure()
	plt.suptitle('Feature_{}_{}'.format(m,n))
	for i in range(len(pred)):
		plt.scatter([features[i][0]], [features[i][1]], c = color[pred[i]], s = 3)
	plt.savefig('cluster_{}_{}'.format(m, n))
	plt.close()
	'''


cluster_num = train(feature_path, cluster_num, weights) + 1
pred(feature_path, cluster_num, 50, weights)

'''
for m in range(11):
	for n in range(11):
		if m != n:
			print("Doing {},{}".format(m,n))
			weights[m] += 5
			weights[n] += 5
			cluster_num = train(feature_path, cluster_num, weights) + 1
			pred(feature_path, cluster_num, 900, weights, m, n)
			weights[m] -= 5
			weights[n] -= 5
'''

'''
data = pd.read_csv(feature_path)
filename_list = np.array(data['id'])
index = np.random.randint(0,len(filename_list),25)
for i in range(len(index)):
	filename = filename_list[index[i]]
	filename = filename[:-4] + '_season.csv'
	d = pd.read_csv(filename)
	value = np.array(d['value'])
	plt.subplot(5,5,i+1)
	plt.plot(value)
plt.show()
'''

