import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import math
from math import floor, ceil, exp

file_sizes = (173669, 275487, 1197613, 1549805, 502334,
	217684, 1796841, 274708, 631252, 148665, 150254, 4784408,
	344759, 440109, 4198037, 329673, 28602, 144173, 1461469,
	187895, 369313, 959307, 1482335, 2772513, 1313997, 254845,
	 486167, 2667146, 264004, 297223, 94694, 1757457, 576203,
	 8577828, 498382, 8478177, 123575, 4062389, 3001419, 196884,
	 617991, 421056, 3017627, 131936, 1152730, 2676649, 656678,
	 4519834, 201919, 56080, 2142553, 326263, 8172117, 2304253,
	 4761871, 205387, 6148422, 414559, 2893305, 2158562, 465972,
	 304078, 1841018, 1915571)
Memory_size = 2**26
T0 = 2**25
alpha = 0.95
iterations = 100000
hmin = 1
hmax = 64




def hamming_distance(i, x):

	h = ceil(-64/(iterations - 1)*i + 64)
	new_x = x.copy()
	for i in range(h):
		ind = random.randint(64)
		new_x[ind] = 1 - new_x[ind]
	return new_x

def loss_function(x):
	sum = Memory_size
	for i in range(len(x)):
		sum -= x[i] * file_sizes[i]

	if(sum<0):
		return Memory_size
	else:
		return sum

def annealing():
	T = T0
	x =  random.randint(2, size=(64))

	loss = loss_function(x)

	trace = []

	for i in range(iterations):

		new_x = hamming_distance(i, x)

		new_loss = loss_function(new_x)

		diff = new_loss - loss


		trace.append(loss)
		if diff < 0 :
			x = new_x
			loss = new_loss
		elif diff == 0:
			if random.rand() < 0.5:
				x = new_x
				loss = new_loss
		elif random.rand() <= exp(-diff / T):
			x = new_x
			loss = new_loss

		T = alpha * T

	return trace,loss,x

min_loss = 99999
min_x = []

xx = []
traces = []

repeat = 20

cumulative_minimum = []


for i in range(repeat):

	print(" {}-th SIMULATION STARTED ".format(i+1))

	trace,loss,x = annealing()
	print(loss)
	if loss < min_loss:
		min_loss = loss
		min_x = x
	xx.append(x)

	traces.append(trace)

	cumulative_minimum.append(np.minimum.accumulate(trace))

print("Minimal loss : {}".format(min_loss))
print(min_x)


median_loss = []

for i in range(iterations):
	sum = 0
	for j in range(repeat):
		sum += cumulative_minimum[j][i]
	median_loss.append(sum/repeat)



print(median_loss)

plt.figure(11)
plt.yscale("log")
plt.xscale("log")

for i in range(repeat):
    plt.plot(np.linspace(0, floor(iterations - 1), floor(iterations)), cumulative_minimum[i])


plt.figure(12)
plt.yscale("log")
plt.xscale("log")
plt.plot(np.linspace(0, floor(iterations - 1), floor(iterations)),median_loss)


plt.show()
