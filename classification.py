from tsfresh.feature_extraction import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from tsfresh.feature_extraction.feature_calculators import skewness, kurtosis, variance, mean, count_above_mean, longest_strike_above_mean, median, absolute_sum_of_changes
from curve_classify import pre_process, normalize
import numpy as np 
import pandas as pd 
import math
import os
import matplotlib.pyplot as plt 
from sklearn.cluster import Birch

'''
funclist = {
	'skewness': None,
	'kurtosis': None,
	'variance': None,
	'mean': None,
	'count_above_mean': None,
	'longest_strike_above_mean': None,
	'median': None,
	'absolute_sum_of_changes': None,
}
'''

def cnt_above_mean(x):
	return count_above_mean(x) / len(x)
def lgst_above_mean(x):
	return longest_strike_above_mean(x) / len(x)
def abs_change(x):
	return absolute_sum_of_changes(x) / len(x)
def cnt_turn(x):
	cnt = 0.0
	for i in range(1, len(x)-1):
		if (x[i]-x[i-1]) * (x[i+1]-x[i]) < 0:
			cnt += 1
	return cnt / len(x)
def cnt_anomaly(x):
	x = np.array(x)
	mean, sigma = np.mean(x), np.std(x)
	cnt = 0
	for i in x:
		if abs(i - mean) > 3 * sigma:
			cnt += 1
	return cnt / len(x)
def amplitude(x):
	amp = max(x) - min(x)
	return amp / np.std(x)

feature_func = [skewness, kurtosis, variance, abs_change, cnt_turn, cnt_anomaly, amplitude]
feature_name = ['skewness', 'kurtosis', 'variance', 'abs_change', 'cnt_turn', 'cnt_anomaly', 'amplitude']
keylist = [3,4,5]
model_path = 'trained_model2.pkl'
feature_path = 'features_new.csv'



def shuffle():
	filename_list = []
	data = pd.read_csv(feature_path)
	model = joblib.load(model_path)
	for i in range(len(data)):
		feature = data.iloc[i]
		filename = feature[0]
		feature = np.array(feature[1:])
		pred = model.predict([feature])
		ok = 1
		for j in keylist:
			if pred == j:
				ok = 0
		if ok:
			filename_list.append(filename)
	return filename_list

def get_average(ts, T = 288):
	N = int(len(ts) / T)
	ans = ts[:T]
	for i in range(1, N):
		ans = ans + ts[i * T: (i + 1) * T]
	ans = ans / N
	return ans 

def get_feature_byvalue(ts):
	ts = np.array(ts)
	feature = []
	for f in feature_func:
		feature.append(f(ts))
	return feature

def get_feature(filename):
	data = pd.read_csv(filename)
	value = np.array(data['value'])
	feature_1 = get_feature_byvalue(value)
	feature = pd.DataFrame()
	feature['id'] = [filename]
	for i in range(len(feature_name)):
		feature[feature_name[i]] = [feature_1[i]]
	print('Finish {}\n'.format(filename))
	print(feature)
	return feature

def get_feature_byts(value, filename):
	feature = pd.DataFrame()
	feature['id'] = [filename]
	for f, v_name in zip(feature_func, feature_name):
		feature[v_name] = [f(value)]
	print('Finish {}\n'.format(filename))
	print(feature)
	return feature

def f(ts):
	T = 288
	ts = np.array(ts)
	ts = pre_process(ts, 200)
	std = holtWinters_forclass(ts)
	std = normalize(std)
	N = int(len(ts) / T)
	ans = 0
	for i in range(0, N):
		for j in range(0, T):
			ans += abs(ts[i * T + j] - std[j])
	ans = ans / (N * T)
	print(ts)
	print(std)
	print(ans)
	return ans

def add_feature(feature_path, f, f_name):
	feature = pd.read_csv(feature_path)
	add_list, cnt = [], 0
	for filename in feature['id']:
		print("Processing {}, which is {}".format(filename, cnt))
		data = pd.read_csv(filename[:-4] + '_season.csv')
		value = np.array(data['value'])
		add_list.append(f(value))
		cnt += 1
	print(add_list)
	feature[f_name] = np.array(add_list)
	feature.to_csv('features_season.csv')

def feature_list(filename_list):
	features = pd.DataFrame()
	cnt = 1
	for filename in filename_list:
		feature = get_feature(filename)
		features = pd.concat([features, feature])
		features.to_csv('features.csv')
		print('Now: {}'.format(cnt))
		cnt += 1
	return features 

if __name__ == '__main__':
	#add_feature(feature_path, amplitude, 'amplitude')
	#filename_list = os.listdir()[2:-1]
	#feature_list(filename_list)
	pass