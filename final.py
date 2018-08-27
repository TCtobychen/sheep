from sklearn.externals import joblib
from curve_classify import pre_process, normalize, get_class
import pandas as pd 
import matplotlib.pyplot as plt 
import math
import numpy as np 
import os
from statsmodels.tsa.seasonal import seasonal_decompose
from feature_method import *

def get_mul(season):
	diff, mul = [], []
	for i in range(288):
		diff.append(abs(season[i+1] - season[i]))
	diff = np.array(diff)
	mean = np.mean(diff)
	for i in range(288):
		if diff[i] > mean:
			mul.append(diff[i] - mean + 3)
		else:
			mul.append(3)
	return mul

def find_time(key, now, ts, sigma, diff_mean, _inc):
	cnt = 0
	while (key -ts[(now-1) % 288] - diff_mean[(now-1) % 288] - 3 * sigma[(now-1) % 288]) *(key - ts[(now+1)%288] - diff_mean[(now+1) % 288]+ 3 * sigma[(now+1) % 288])> 0 and (key -ts[(now-1) % 288] - diff_mean[(now-1) % 288]+ 3 * sigma[(now-1) % 288]) *(key - ts[(now+1)%288]-diff_mean[(now+1) % 288] - 3 * sigma[(now+1) % 288])> 0 and cnt < 30:
		now += _inc
		cnt += 1
	return cnt

def _getanomaly_graph(anomaly, value, key = 1):	
	# base is used to locate the starting point on the final graph. upshift is used to make the final graph more clear.  
	anomaly_x = []
	anomaly_y = []
	for i in range(0, len(anomaly)):
		if anomaly[i] == key:
			anomaly_x.append(i)
			anomaly_y.append(value[i])
	return anomaly_x, anomaly_y

def get_time_diff(ts):
	diff, season, abs_value, trend = get_seasonal_diff(ts, _normalize = False)
	res, sigma = [], np.std(diff)
	for i in range(len(ts)):
		res.append(min(find_time(ts[i], i%288, season + trend, sigma, diff_mean,1), find_time(ts[i], i%288, season + trend, sigma, diff_mean, -1)))
	return res

def anomaly_stable_diff(ts):
	mul = 2
	diff, season, abs_value, trend = get_seasonal_diff(ts, _normalize = False)
	value = pre_process(ts, _normalize = False)
	sigma, diff_mean = [], []
	N = len(ts) // 288
	for j in range(288):
		temp = []
		for i in range(N):
			temp.append(value[i * 288 + j] - season[j] - trend)
		temp = sorted(temp, key = abs)[:-len(temp)//3]
		sigma.append(np.std(temp))
		diff_mean.append(np.mean(temp))
	'''
	time_diff = get_time_diff(ts)
	plt.figure()
	plt.plot(time_diff)
	plt.show()
	plt.close()
	'''
	anomaly = []
	for i in range(len(ts)):
		if abs(ts[i] - season[i % 288] - trend - diff_mean[i % 288]) > mul * sigma[i % 288]:
			if find_time(ts[i], i%288, season + trend, sigma,diff_mean, 1) > 9 and find_time(ts[i], i % 288, season + trend, sigma,diff_mean, -1) > 9:
				anomaly.append(1)
			else:
				anomaly.append(0)
		else:
			anomaly.append(0)
	anomaly_x, anomaly_y = _getanomaly_graph(anomaly, ts)
	plt.figure()
	plt.scatter(anomaly_x, anomaly_y, marker = 'x', s = 10, c = 'red')
	plt.plot(ts)
	plt.title('Stable')
	plt.plot(season + trend, linewidth = 1)
	#plt.plot(season[:288]+trend + diff_mean + sigma+ sigma+ sigma)
	#plt.plot(season[:288]+trend + diff_mean - sigma- sigma- sigma)

def anomaly_sharp_diff(ts):
	mul = 3
	diff, season, abs_value, trend = get_seasonal_diff(ts, _normalize = False)
	value = pre_process(ts, _normalize = False)
	sigma, diff_mean = [], []
	N = len(ts) // 288
	for j in range(288):
		temp = []
		for i in range(N):
			temp.append(value[i * 288 + j] - season[j] - trend)
		temp = sorted(temp, key = abs)[:-len(temp)//3]
		sigma.append(np.std(temp))
		diff_mean.append(np.mean(temp))
	'''
	time_diff = get_time_diff(ts)
	plt.figure()
	plt.plot(time_diff)
	plt.show()
	plt.close()
	'''
	anomaly = []
	for i in range(len(ts)):
		if abs(ts[i] - season[i % 288] - trend - diff_mean[i % 288]) > mul * sigma[i % 288]:
			if find_time(ts[i], i%288, season + trend, sigma,diff_mean, 1) > 12 and find_time(ts[i], i % 288, season + trend, sigma,diff_mean, -1) > 12:
				anomaly.append(1)
			else:
				anomaly.append(0)
		else:
			anomaly.append(0)
	anomaly_x, anomaly_y = _getanomaly_graph(anomaly, ts)
	plt.figure()
	plt.scatter(anomaly_x, anomaly_y, marker = 'x', s = 10, c = 'red')
	plt.plot(ts)
	plt.title('Sharp')
	plt.plot(season + trend, linewidth = 1)

def anomaly_safe(ts):
	plt.plot(ts)
	plt.title('Safe')

def combine(feature, weights):
	if len(feature) != len(weights):
		print("Danger!!! weights' length is not right!\n\n")
		return
	ans = []
	for i in range(len(weights)):
		if weights[i] != 0:
			ans.append(weights[i] * feature[i])
	return ans

def get_diff_type(ts):
	diff, season, abs_value, trend = get_seasonal_diff(ts)
	features = get_feature_byvalue(diff)
	weights1 = [0,0,0,0,0,5,0]
	weights2 = [0,0,0,0,0,0,5]
	model1 = joblib.load('diff_model_anomalycnt.pkl')
	model2 = joblib.load('diff_model_amplitude.pkl')
	pred1 = model1.predict([combine(features, weights1)])
	pred2 = model2.predict([combine(features, weights2)])
	if pred1 == 0 and pred2 == 1:
		pred = 0
	else:
		pred = 1
	return pred, (diff, season, trend) 

def get_daykind_curve(filename, value_index):
	#if get_class(filename) == 1:
	#	print("Not daykind curve !!!! Cannot Handle! \n\n")
	#	return
	data = pd.read_csv(filename)
	value = np.array(data[data.columns[value_index]])
	key = Bf_period(pre_process(value))
	if key <= 0.005:
		c = 0
	if key > 0.005 and key <= 0.008:
		c = 1
	if key > 0.008:
		c = 2
	res = anomaly_detect_func[c](value)
	plt.show()
	#plt.savefig(filename[:-4], dpi = 800)
	#plt.close()

def anomaly_stupid_diff(ts):
	print("Not Periodic!!!!\n\n")

anomaly_detect_func = [anomaly_stable_diff, anomaly_sharp_diff, anomaly_safe]

if __name__ == '__main__':
	data = pd.read_csv('featuresnew.csv')
	filename_list = np.array(data['id'])
	for filename in filename_list[:4]:
		get_daykind_curve(filename, 1)