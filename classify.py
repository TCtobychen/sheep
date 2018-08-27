from sklearn.externals import joblib
from curve_classify import pre_process, normalize, get_feature
import pandas as pd 
import matplotlib.pyplot as plt 
import math
import numpy as np 
import os
from statsmodels.tsa.seasonal import seasonal_decompose

cluster_num = 15
'''
data = pd.read_csv('features_withid.csv')
features, filename_list = [], []
for i in range(len(data[data.columns[1]])):
	feature = np.array(data.iloc[i])
	filename_list.append(feature[0])
	features.append(feature[1:])
clf = joblib.load('trained_model2.pkl')
pred = clf.predict(features)
'''

def visual(filename_list, pred, cluster_num, index = 1, Length = 3000):
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
			value = np.array(data[data.columns[index]])
			value = pre_process(value)
			plt.subplot(N, N, i + 1)
			plt.xticks([])
			plt.yticks([])
			plt.plot(value)
		plt.savefig('Figure_for_Class{}'.format(ind), dpi = 600)
		ind += 1

def visual_withseason(filename_list, pred, cluster_num, index = 1, Length = 3000):
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
			value = np.array(data[data.columns[index]])
			value = pre_process(value)
			plt.subplot(N, N, i + 1)
			plt.xticks([])
			plt.yticks([])
			plt.plot(value, linewidth = 0.5)
			diff, season, abs_value, trend = get_seasonal_diff(value)
			plt.plot(season + trend, linewidth = 0.5)
		plt.savefig('Figure_for_Class{}'.format(ind), dpi = 1000)
		ind += 1

def visual_justdiff(filename_list, pred, cluster_num, index = 1, Length = 3000):
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
			print(filename_list[L[i]])
			if len(data) > Length:
				data = data[:Length]
			value = np.array(data[data.columns[index]])
			#value = pre_process(value)
			plt.subplot(N, N, i + 1)
			axes = plt.gca()
			axes.set_ylim([-0.5, 0.5])
			plt.xticks([])
			plt.yticks([])
			plt.plot(value, linewidth = 0.5)
		plt.savefig('Figure_for_Class{}_diff'.format(ind), dpi = 1000)
		ind += 1
		plt.close()

def get_abs_diff(ts, season, trend):
	ans, diff = 0, []
	for i in range(len(ts)):
		diff.append(ts[i] - season[i] - trend)
		ans += abs(diff[i])
	return (ans, diff)

def get_seasonal_diff(ts, T = 288, _normalize = True):
	# return best fitted seasonal_diff, season series, absolute diff and the average trend used. 
	# ----------------------------------------------------------------------------
	ts = np.array(ts)
	ts = pre_process(ts, _normalize = _normalize)
	res = seasonal_decompose(ts, freq = T)
	start, end = min(ts), max(ts)
	season = res.seasonal
	_abs, diff = [], []
	for j in range(100):
		temp0, temp1 = get_abs_diff(ts, season, start + j * (end - start) / 100)
		_abs.append(temp0)
		diff.append(temp1)
	ind, _min = 0, len(ts) * max(ts)
	for i in range(len(_abs)):
		if _abs[i] < _min:
			_min = _abs[i]
			ind = i
	return (diff[ind], season, _abs[ind], start + (ind - 1) * (end - start) / 100)

def stl_decompose(ts):
	res = seasonal_decompose(ts, freq = 288)
	for i in range(0,3):
		print(res.seasonal[i * 288: (i + 1) * 288])
	print(len(res.seasonal))
	plt.show(res.plot())
	plt.figure()
	plt.plot(pre_process(ts))
	plt.plot(normalize(res.seasonal))
	plt.show()

'''
data = pd.read_csv('_0_4_28_66827.csv')
value = np.array(data['value'])
diff, season, abs_value, trend = get_seasonal_diff(value)
plt.plot(pre_process(value))
plt.plot(diff)
plt.plot(season + trend)
print(trend)
plt.show()
'''

def get_class(filename, value_index = 1):
	model_1 = joblib.load('trained_model2.pkl')
	model_2 = joblib.load('trained_model3.pkl')
	data = pd.read_csv(filename)
	value = np.array(data[data.columns[value_index]])
	feature = get_feature(filename)

# 这个是模型训练完，用来看看是不是能够实现粗糙分类的函数