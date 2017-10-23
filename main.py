import Tkinter
import pandas as pd 
from matplotlib import pyplot as plt
from numpy import average as avg
from numpy import exp as exp
from numpy import linspace as linspace
from scipy.optimize import minimize
from math import ceil

# Read data 
history = pd.read_csv('./../data/04_cricket_1999to2011.csv')

# Ignore corrupt entries
def find_corrupt_matches():
	corrupt = []
	for i, match in enumerate(history['Match']):
		if i == 0:
			continue
		if history['Over'][i] == 1 and history['Innings'][i] == 2 and history['Wickets.in.Hand'][i-1] != 0\
		and history['Over'][i-1] != 50: # Unfinished 1st innings
			corrupt.append(match)
		if history['Over'][i] == 1 and history['Runs.Remaining'][i-1] != 0:
		# Unfinished 1st or 2nd innings
			corrupt.append(match)
	return corrupt

corrupt = find_corrupt_matches()
 
# Find number of overs played in an innings
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

# Create observation table for Z(u,w)
def create_table():
	table = []
	d = {}
	for over_rem in range(50):
		for wckt_rem in range(11):
			d[(over_rem, wckt_rem)] = []
	for i, match in enumerate(history['Match']):
		if match not in corrupt:
			w = history['Wickets.in.Hand'][i]
			t = history['Runs.Remaining'][i]
			over = history['Over'][i]
			if history['Innings'][i] == 1:
				u = 50 - over
			else:
				u = op[i] - over
			if u == 0 or w == 0:
				t = -1
			d[(u, w)].append(t)
	for over_rem in range(50):
		row = []
		for wckt_rem in range(11):
			if len(d[(over_rem, wckt_rem)]) >= 30: # If at least 25 data points aren't found we ignore the
				row.append(ceil(avg(d[(over_rem, wckt_rem)]))) # situation while optimizing
			else:
				row.append(-1)
		table.append(row)
		#print table[over_rem]
	return table


table = create_table()

# Run production function
def prod(u, Z0, L):
	return Z0*(1-exp(-L*u/Z0))

# Resource function
def P(u, Z0, Z, L):
	return prod(u, Z0, L)/prod(50, Z, L)

# Loss function for optimization
def loss(Z): 
	s = 0
	for over_rem in range(1,50):
		for wckt_rem in range(1, 11):
			t = table[over_rem][wckt_rem]
			if t != -1:
				s = s + (prod(over_rem, Z[wckt_rem-1], Z[10]) - t)**2
	return s

# Parameter calculation
x0 = []
bnds = []
for i in range(10):
	x0.append((i+1)*50)
	bnds.append((1, None))
x0.append(5)
bnds.append((0,None))
params = minimize(loss, x0, bounds=bnds).x
print('############ Parameters ############\n')
for i in range(10):
	print('Z({}) = {}'.format(i+1, params[i]))
print('L = {}'.format(params[10]))

# Plot fraction of resources remaining
fig, ax = plt.subplots(figsize=(8,5))
X = linspace(0, 50, 100)
for i in range(10):
	Y = P(X, params[i], params[9], params[10])
	ax.plot(X, Y, label = str(i+1))
	ax.set_xlabel('overs remaining')
	ax.set_ylabel('fraction of resources remaining')
plt.legend()
plt.savefig('resource' + '.png')

# Calculate error in Z(u,w) when w is fixed and plot it
def error(w,):
	s, d = 0, 0
	for over in range(1, 50):
		t = table[over][w]
		if t != -1:
			s = s + (prod(over, params[w-1], params[10])-t)**2
			d = d + t**2
	fig, ax = plt.subplots(figsize = (8,5))
	X, Y = [], []
	for i in range(50):
		t = table[i][w]
		if t != -1:
			X.append(i)
			Y.append(t)
	ax.scatter(X,Y, color = 'red')
	Y = [prod(x, params[w-1], params[10]) for x in X]
	ax.plot(X, Y, color = 'blue')
	ax.set_xlabel('overs remaining')
	ax.set_ylabel('average runs scored in remaining overs')
	plt.savefig('wickets_in_hand__'+ str(w) + '.png')
	return 100*(s/d)**0.5

# Calculate error in Z(u,w)
def total_error():
	s = 0
	for over_rem in range(1,50):
		for wckt_rem in range(1, 11):
			t = table[over_rem][wckt_rem]
			if t != -1:
				s = s + t**2
	return 100*loss(params)/s	

print('\n############ Errors ############\n')
for w in range(1,11):
	print('Relative error in Z(u,{}) is {}%'.format(w, error(w)))
print('Relative error in Z(u,w) is {}%'.format(total_error()))