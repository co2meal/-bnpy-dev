'''
ToyBinaryK4.py


'''
import numpy as np
import matplotlib.pyplot as plt

SEED = 8943337
PRNG = np.random.RandomState(SEED)

# FIXED DATA GENERATION PARAMS
K = 2 # number of topics

def get_data_info():
  s = 'Toy Binary Data with %d topics.' % (K)
  return s

def get_data():
	pass

def generateSample(p, size):
	return np.random.binomial(1, p, size)

def visualizeData(data, size):
	width = .7
	ind = np.arange(size)
	fig, ax = plt.subplots()
	rect = ax.bar(ind, sum(data, 1), width, color='y')

	ax.set_ylabel('Counts')
	ax.set_xticks(ind+width/2)
	ax.set_xticklabels(ind)

	plt.show()

def em(observations, prior, cont_tol, iterations):
	iteration = 0
	while iteration < iterations:
		# E step
		# M step
		if delta_change < cont_tol:
			break
		else:
			iteration += 1
	return [post_est, iteration]


def main():
	coins = [.8, .5] 
	num_sample = 100

	num_tosses = [3, 2] # biased three times, fair twice
	tosses = [coins[0]]*num_tosses[0] + [coins[1]]*num_tosses[1]
	np.random.shuffle(tosses) # random shuffle the coin tosses

	observations = generateSample(tosses, [num_sample, len(tosses)])

	##### vis
	print 'true coins probability', coins
	print 'observations', sum(observations, 1)
	visualizeData(observations, len(tosses))
	#####

if __name__ == '__main__':
	main()
