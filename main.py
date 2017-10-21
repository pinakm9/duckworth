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

def set_overs_played():
	op_, start, overs = [], 0, history['Over']
	for i,over in enumerate(overs):
		if i == 0:
			continue
		if over == 1:
			played = overs[i-1]
			op_.append([played]*(i-start))
			start = i
	op, l = [], len(overs)
	op_.append([overs[l-1]]*(l-start))
	for sublist in op_:
		for item in sublist:
			op.append(item)
	return op


op = set_overs_played()

def create_table():
	table = []
	d = {}
	for over_rem in range(50):
		for wckt_rem in range(11):
			d[(over_rem, wckt_rem)] = []
	for i, over in enumerate(history['Over']):
		if history['Innings'][i] == 1:
			w = history['Wickets.in.Hand'][i]
			d[(op[i]-over, w)].append(history['Runs.Remaining'][i])
	for over_rem in range(50):
		row = []
		for wckt_rem in range(11):
			if d[(over_rem, wckt_rem)] != []:
				row.append(ceil(avg(d[(over_rem, wckt_rem)])))
			else:
				row.append(-1)
		table.append(row)
		print table[over_rem]
	return table


table = create_table()

def prod(u, Z0, L):
	return Z0*(1-exp(-L*u/Z0))

def P(u, Z0, Z, L):
	return prod(u, Z0, L)/prod(50, Z, L)


def loss(Z): 
	s = 0
	for over_rem in range(50):
		for wckt_lost in range(11):
			t = table[over_rem][wckt_lost]
			if t != -1:
				s = s + (prod(over_rem, Z[wckt_lost-1], Z[10]) - t)**2
	return s

x0 = []
bnds = []
for i in range(10):
	x0.append(i*30)
	bnds.append((10, 5000))#(i*30,(i+1)*30))
x0.append(5)
bnds.append((0,None))
res = minimize(loss, x0, bounds=bnds)#, tol = 1e-6)
print res

fig, ax = plt.subplots(figsize=(8,5))
X = np.linspace(0, 50, 100)
for i in range(10):
	Y = P(X, res.x[i], res.x[9], res.x[10])
	ax.plot(X, Y)
plt.savefig('img_name' + '.png')

fig, ax = plt.subplots(figsize=(8,5))
X = np.linspace(0, 50, 100)
for i in range(10):
	Y = P(X, res.x[i], res.x[9], res.x[10])
	ax.plot(X, Y)
plt.savefig('img_name' + '.png')

print(loss(res.x))


def error(w,):
	s, d = 0, 0
	for over in range(50):
		t = table[over][w]
		if t != -1 and t != 0:
			s = s + (prod(over, res.x[w-1], res.x[10])-t)**2
			d = d + (t)**2
	fig, ax = plt.subplots(figsize=(8,5))
	X = list(range(50))
	Y = [table[x][w] for x in X]
	ax.scatter(X,Y)
	Y = [prod(x, res.x[w-1], res.x[10]) for x in X]
	ax.plot(X, Y)
	plt.show()
	return [d**0.5, 100*(s/d)**0.5]


print error(6)