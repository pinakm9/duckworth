import Tkinter
import pandas as pd 
from matplotlib import pyplot as plt
import numpy as np
from numpy import average as avg
from numpy import exp as exp
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from numpy import array
from math import ceil

history = pd.read_csv('./../data/04_cricket_1999to2011.csv')

def find_total(data, wickets, overs):
	total = []
	l = len(data['Over'])
	for i, over in enumerate(data['Over']):
		if over == 50-overs and data['Wickets.in.Hand'][i] == wickets:
			total.append(data['Innings.Total.Runs'][i])
	return total


def create_table():
	table = []
	d = {}
	for over_rem in range(50):
		for wckt_lost in range(10):
			d[(over_rem, wckt_lost)] = []
	for i, over in enumerate(history['Over']):
		w = history['Wickets.in.Hand'][i]
		if w > 0:
			d[(50-over, 10-w)].append(history['Runs.Remaining'][i])
	for over_rem in range(50):
		row = []
		for wckt_lost in range(10):
			if d[(over_rem, wckt_lost)] != []:
				row.append(ceil(avg(d[(over_rem, wckt_lost)])))
			else:
				row.append(-1)
		table.append(row)
	return table


table = create_table()
for over_rem in range(50):
		print(over_rem, table[over_rem])

def set_data(data, wickets):
	xdata, ydata, rate = [], [], []
	for overs in range(1, 51):
		runs = find_total(data, wickets, overs)
		if runs != []:
			xdata.append(overs)
			ydata.append(avg(runs))
			l = len(xdata)-1
			rate.append(ydata[l]/xdata[l])
	return [xdata, ydata, rate]

def prod(u, Z0, L):
	return Z0*(1-exp(-L*u/Z0))

def P(u, Z0, Z, L):
	return prod(u, Z0, L)/prod(50, Z, L)

def fit(data, wickets):
	z0, l = curve_fit(prod, *set_data(data, wickets)[:-1])[0]
	return [z0, l]

over = {}
z = {}
for w in range(1, 11):
	over[w], z[w] = set_data(history, w)[0:2]

def loss(Z): 
	s = 0
	for over_rem in range(50):
		for wckt_lost in range(10):
			t = table[over_rem][wckt_lost]
			if t != -1:
				s = s + (prod(over_rem, Z[wckt_lost-1], Z[10]) - t)**2
	return s

x0 = []
bnds = []
for i in range(10):
	x0.append(100)#i*25)
	bnds.append((150,450))
x0.append(0.1)
print x0
bnds.append((0,None))
print x0
res = minimize(loss, x0, bounds=bnds)#, tol = 1e-6)
print res.x

fig, ax = plt.subplots(figsize=(8,5))
X = np.linspace(0, 50, 100)
for i in range(10):
	Y = P(X, res.x[i], res.x[9], res.x[10])
	ax.plot(X, Y)
plt.savefig('img_name' + '.png')

print(loss(res.x))

