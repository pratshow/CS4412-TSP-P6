# Brandon Watkins
# CS 4412

#!/usr/bin/python3
import math
import random

from PriorityQueue import *
from which_pyqt import PYQT_VER
if PYQT_VER == 'PYQT5':
	from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
	from PyQt4.QtCore import QLineF, QPointF
else:
	raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))




import time
import numpy as np
from TSPClasses import *
import heapq
import itertools
from copy import deepcopy



class TSPSolver:
	def __init__( self, gui_view ):
		self._scenario = None
		self._bssf = None

	def setupWithScenario( self, scenario ):
		self._scenario = scenario


	''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution, 
		time spent to find solution, number of permutations tried during search, the 
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''
	
	def defaultRandomTour( self, time_allowance=60.0 ):
		results = {}
		cities = self._scenario.getCities()
		ncities = len(cities)
		foundTour = False
		count = 0
		bssf = None
		start_time = time.time()
		while not foundTour and time.time()-start_time < time_allowance:
			# create a random permutation
			perm = np.random.permutation( ncities )
			route = []
			# Now build the route using the random permutation
			for i in range( ncities ):
				route.append( cities[ perm[i] ] )
			bssf = TSPSolution(route)
			count += 1
			if bssf.cost < np.inf:
				# Found a valid route
				foundTour = True
		end_time = time.time()
		results['cost'] = bssf.cost if foundTour else math.inf
		results['time'] = end_time - start_time
		results['count'] = count
		results['soln'] = bssf
		results['max'] = None
		results['total'] = None
		results['pruned'] = None
		return results


	''' <summary>
		Greedy algorithm for TSP. Cycles through all of the cities, using each as a starting city. At each city, this
		greedily selects the city with the cheapest cost to visit next. The best route discovered is kept/returned.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

	def greedy(self, time_allowance=60.0):
		results = {}
		intermediateSolutions = 0
		startTime = time.time()
		cities = self._scenario.getCities()
		nCities = len(cities)
		bssf = None

		# Cycling through all cities, setting the starting city to each
		for i in range(nCities):
			if time.time() - startTime > time_allowance: break

			# May as well use the B&B states.
			# They do what we need to, with some additional overhead (Don't really need to update the cost matrix).
			# If you were to run this more than just n_cities times, you'd want to swap it out for something more efficient.
			base = State(partialPath=cities, startingCityIndex=i)

			# Just picks the city with the cheapest cost, until no legal edges remain
			while len(base.partialPath) < len(cities):
				children = base.expand()
				if len(children) == 0: break
				children.sort(key=lambda c: c.bound)
				base = children[0]
			# If the route was complete, legal, and an improvement, update the bssf
			if base.isValidSolution():
				if bssf is not None:
					if base.lowerbound() < bssf.lowerbound():
						intermediateSolutions += 1
						bssf = base
				else: bssf = base

		end_time = time.time()

		self._bssf = TSPSolution(getCityPath(cities, bssf)) if bssf is not None else None

		results['cost'] = self._bssf.cost if self._bssf is not None else math.inf
		results['time'] = end_time - startTime
		results['count'] = intermediateSolutions
		results['soln'] = self._bssf if self._bssf is not None else None
		results['max'] = "--"
		results['total'] = "--"
		results['pruned'] = "--"

		return results

	
	''' <summary>
		Branch and bound optimal algorithm.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints: 
		max queue size, total number of states created, and number of pruned states.</returns> 
	'''

	def branchAndBound(self, time_allowance=60.0):
		results = {}
		cities = self._scenario.getCities()
		n_cities = len(cities)
		intermediateSolutions = 0
		totalPruned = 0
		maxQueueSize = 0
		totalChildStatesGenerated = 0
		start_time = time.time()

		self._bssf = self.initialBSSF(cities=cities, attempts=n_cities ** 2, time_allowance=time_allowance / 5)

		baseProblem = State(partialPath=cities)
		S = DAryHeapPriorityQueue(d=n_cities, elementList=[baseProblem])
		while S.notEmpty() and time.time() - start_time < time_allowance:
			maxQueueSize = max(maxQueueSize, len(S.queue))
			subproblem = S.deleteMin()
			if subproblem.lowerbound() == math.inf or (self._bssf is not None and subproblem.lowerbound() > self._bssf.lowerbound()):
				totalPruned += 1
				continue
			smallerSubproblems = subproblem.expand()
			totalChildStatesGenerated += len(smallerSubproblems)

			# Unsure if I'm supposed to prune cities inside expand(). Also unsure if I'm supposed to include them in totalPruned
			totalPruned += (n_cities - len(subproblem.partialPath) - len(smallerSubproblems))

			for P in smallerSubproblems:
				if P.lowerbound() == math.inf: totalPruned += 1
				elif P.isValidSolution():
					if self._bssf is None or P.lowerbound() < self._bssf.lowerbound():
						self._bssf = P
						intermediateSolutions += 1
				elif self._bssf is None or P.lowerbound() < self._bssf.lowerbound():
					# Scales the priority, so that initially it favors a "deep" search, gradually transitioning to wide
					S.insert(P, P.lowerbound() * time_allowance / (P.partialPathSize() * (time_allowance - (time.time() - start_time))))
				else:
					totalPruned += 1

		maxQueueSize = max(maxQueueSize, len(S.queue))
		totalPruned += len(S.queue)
		end_time = time.time()

		self._bssf = TSPSolution(getCityPath(cities, self._bssf)) if self._bssf is not None else None

		results['cost'] = self._bssf.cost if self._bssf is not None else math.inf
		results['time'] = end_time - start_time
		results['count'] = intermediateSolutions
		results['soln'] = self._bssf
		results['max'] = maxQueueSize
		results['total'] = totalChildStatesGenerated
		results['pruned'] = totalPruned

		return results


	''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''
		
	def fancy(self, time_allowance=60.0):
		pass


	'''
		Attempts to find an initial BSSF via an incredibly simple greedy algorithm, followed by checking random routes
		for the given number of attempts. Returns the best route's state, or None if none found.
	'''

	def initialBSSF(self, cities, attempts = 3, time_allowance = 10.):
		bssf = None
		start_time = time.time()

		# Just checking for a quick greedy route.
		base = State(partialPath=cities)
		while len(base.partialPath) < len(cities):
			children = base.expand()
			if len(children) == 0: break
			children.sort(key=lambda c: c.bound)
			base = children[0]
		if base.isValidSolution():
			bssf = base

		# Checking random routes for the specified number of attempts
		count = 0
		indices = [i for i in range(1, len(cities))]
		while time.time()-start_time < time_allowance and count < attempts:
			base = State(partialPath=cities)
			random.shuffle(indices)

			for j in range(len(indices)):
				if base.lowerbound() == math.inf or base.isValidSolution(): break
				children = base.expand()
				if len(children) == 0: break
				for child in children:
					if child.id == indices[j]:
						base = child
						break

			if base.isValidSolution():
				if bssf is None or base.lowerbound() < bssf.lowerbound(): bssf = base
			count += 1

		# Ran out of time, or attempts. Returning best solution, or None if none found
		return bssf


'''
	Computes the new lower bound for the given matrix, and reduces the cost matrix.
	Modifies the input State's costMatrix and lowerBound, in place.
'''

def lowerBound(state):
	assert(type(state) == State)
	# Reduce cost matrix rows
	# This represents taking the cheapest edge FROM each of the nodes.
	# Reducing the other values by this amount, because taking those edges instead would result in their cell's cost,
	# in addition to the cheapest.
	for i in range(len(state.costMatrix[0])):
		cheapest = min(state.costMatrix[i])
		if cheapest == math.inf:
			if not state.partialPath.__contains__(i): state.bound = math.inf
			continue
		state.bound += cheapest
		for j in range(len(state.costMatrix[i])):
			state.costMatrix[i][j] -= cheapest

	# Reduce cost matrix columns
	# This represents taking the cheapest edge TO each of the nodes.
	for j in range(len(state.costMatrix)):
		cheapest = min(row[j] for row in state.costMatrix)
		if cheapest == math.inf:
			if not state.partialPath.__contains__(j): state.bound = math.inf
			continue
		state.bound += cheapest
		for i in range(len(state.costMatrix[j])):
			state.costMatrix[i][j] -= cheapest


'''
	Computes/stores the cost to go from every node to every other node, and stores costs in a matrix.
	Note: This does not reduce the matrix at all.
'''

def initialCostMatrix(cities):
	# Create initial cost matrix
	costMatrix = []
	for i in range(len(cities)):
		inner = []
		for j in range(len(cities)):
			inner.append(math.inf)
		costMatrix.append(inner)

	# Fill out cost matrix
	i = 0
	j = 0
	for city in cities:
		for innerCity in cities:
			if city != innerCity:
				costMatrix[i][j] = city.costTo(innerCity)
			j += 1
		j = 0
		i += 1

	return costMatrix


'''
	This is what "visits" the next node, adjusting the bound, and adding the new math.inf row/col.
	The cost matrix still needs to be reduced.
'''

def updateCostMatrix(state = None):
	assert type(state) == State
	state.bound += state.costMatrix[state.partialPath[-2]][state.id]

	for i in range(len(state.costMatrix[0])):
		state.costMatrix[state.partialPath[-2]][i] = math.inf
		state.costMatrix[i][state.id] = math.inf
	state.costMatrix[state.id][state.partialPath[-2]] = math.inf


'''
	Gets the actual list of cities in order of "visitation" from the travelling salesman (starting city is first/index 0)
'''

def getCityPath(cities, state = None):
	assert(type(state) == State)
	path = state.getPath()
	returnPath = []
	for cityIndex in path:
		returnPath.append(cities[cityIndex])
	return returnPath


'''
	State for storing the cost matrix, lowerbound, and partial path (cities remaining to be visited).
	Also includes some utility functions for extracting child states, automating calculations, and generating full path.
	Note: partialPath originally contains the list of cities, but all children states use this to reference the 
	city index that is about to be evaluated. self.partialPath will contain a list of all city indexes waiting to be evaluated.
'''

class State:

	def __init__(self, parent = None, partialPath = None, startingCityIndex = 0):
		assert(partialPath is not None)
		assert (parent is None or type(parent) == State)
		assert((parent is not None and not parent.partialPath.__contains__(partialPath)) or parent is None)
		self.bound = parent.bound if parent is not None else 0
		# To keep track of which city this is for.
		self.id = partialPath if parent is not None else startingCityIndex # Can probably get rid of this, was used before I fixed my partialPath

		if parent is None:
			self.partialPath = [self.id]
			self.costMatrix = initialCostMatrix(partialPath)
		else:
			self.costMatrix = deepcopy(parent.costMatrix)
			self.partialPath = deepcopy(parent.partialPath)
			self.partialPath.append(partialPath)
			updateCostMatrix(self)

		lowerBound(self)

	def lowerbound(self):
		return self.bound

	def partialPathSize(self):
		return len(self.partialPath)

	def isCompleteSolution(self):
		return len(self.partialPath) == len(self.costMatrix[0])

	def isValidSolution(self):
		return self.isCompleteSolution() and self.costMatrix[self.id][self.partialPath[0]] != math.inf and self.bound != math.inf

	def expand(self):
		children = []
		for i in range(len(self.costMatrix[0])):
			if self.costMatrix[self.id][i] != math.inf and (
					not self.partialPath.__contains__(i) or
					(self.partialPath.index(i) == len(self.partialPath) - 1)
					and self.partialPath.count(i) == 1):
				children.append(State(parent=self, partialPath=i))
		return children

	def getPath(self):
		return self.partialPath

	def toString(self):
		s = ""
		for i in range(len(self.partialPath) - 1):
			s += str(self.partialPath[i]) + ", "
		return s + str(self.partialPath[len(self.partialPath) - 1])