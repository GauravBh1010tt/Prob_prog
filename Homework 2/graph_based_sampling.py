import torch
import torch.distributions as dist

from daphne import daphne
from primitives import *
from tests import is_tol, run_prob_test,load_truth

import warnings
warnings.filterwarnings("ignore")

#from primitives import standard_env

# Put all function mappings from the deterministic language environment to your
# Python evaluation context here:
#env = {'normal': dist.Normal,'sqrt': torch.sqrt,'vector': }

env = standard_env()

env.update({
	#'vector' : lambda *x: torch.tensor(x),
	'vector' : lambda *x: eval(['vector', x]),
	'hash-map': lambda *x: eval(['hash-map', x]),
	})
#print (global_env)

def deterministic_eval(exp):
	"Evaluation function for the deterministic target language of the graph based representation."
	#print ('in', exp)
	#print (env)
	if type(exp) is list:
		op = exp[0]
		args = exp[1:]
		return env[op](*map(deterministic_eval, args))
		#for i in exp:
		#    val = eval(i)
		#return val
	elif type(exp) is int or type(exp) is float:
		# We use torch for all numerical objects in our evaluator
		return torch.tensor(float(exp))
	else:
		raise("Expression type unknown.", exp)


def sample_from_joint(graph):
	"This function does ancestral sampling starting from the prior."
	# TODO insert your code here

	#print (graph[1]['P'])
	search_graph = graph[1]['P']

	#print ('to return', graph[-1])

	ret = {}

	#print (search_graph)

	search = list(search_graph.keys())

	'''
	for i, key in enumerate(search):
		if len(key)<8:
			search[i] = 'sample0'+key[6]
			print (search[i], key)
			search_graph[search[i]] = search_graph.pop(key)
		print (search_graph[key])
		if len(search_graph[key][-1])<8:
			print ('less', search_graph[key])

	search.sort()
	'''

	#print ('here', search)

	search = sorted_alnum(search)

	#print (search)

	for node in search:

		if search_graph[node][0] != 'observe*':
			if search_graph[node][0] == 'sample*':
				search_graph[node][0] = 'sample'
			#print ('topo', node, search_graph[node])
			val = eval(search_graph[node], env)
			#print (val)
			ret[node] = val.item()
			#env.update({    node:val,})
			env[node] = val
			#print ('here', env.find(node))
			#print(node, env[node])
			#print (val)

	if isinstance(graph[-1], list):
		return eval(graph[-1], env)

	return [ret[graph[-1]]]


def get_stream(graph):
	"""Return a stream of prior samples
	Args: 
		graph: json graph as loaded by daphne wrapper
	Returns: a python iterator with an infinite stream of samples
		"""
	while True:
		yield sample_from_joint(graph)


#Testing:

def run_deterministic_tests():
    
	for i in range(1,13):
		#note: this path should be with respect to the daphne path!
		graph = daphne(['graph','-i','/Users/gaurav/Desktop/CS532-HW2/programs/tests/deterministic/test_{}.daphne'.format(i)])
		truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))

		ret = deterministic_eval(graph[-1])
		print ('::Test case::', i, ' ::graph::', graph, '    ::return::', ret)
		try:
			assert(is_tol(ret, truth))
		except AssertionError:
			raise AssertionError('return value {} is not equal to truth {} for graph {}'.format(ret,truth,graph))
		
		print('Test passed')
		
	print('All deterministic tests passed')
	


def run_probabilistic_tests():
	
	#TODO: 
	num_samples=1e5
	max_p_value = 1e-4
	
	for i in range(1,7):
		#note: this path should be with respect to the daphne path!        
		graph = daphne(['graph', '-i', '/Users/gaurav/Desktop/CS532-HW2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
		truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))

		stream = get_stream(graph)
		
		p_val = run_prob_test(stream, truth, num_samples)
		
		print ('::Test case::', i, ' ::graph::', graph, '    ::p-value::', p_val)
		
		assert(p_val > max_p_value)
	
	print('All probabilistic tests passed')    

		
		
if __name__ == '__main__':
	

	run_deterministic_tests()
	run_probabilistic_tests()


	for i in range(1,5):
		graph = daphne(['graph','-i','/Users/gaurav/Desktop/CS532-HW2/programs/{}.daphne'.format(i)])
		print('\n\n\nSample of prior of program {}:'.format(i))
		#print (graph)
		print(sample_from_joint(graph))    

	