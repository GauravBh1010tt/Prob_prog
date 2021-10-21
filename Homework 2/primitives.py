import torch
import operator as op

import re

#TODO

def sorted_alnum(l):
	convert = lambda text: int(text) if text.isdigit() else text
	alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
	return sorted(l, key = alphanum_key)

Symbol = str              
Number = (torch.int32, torch.float32, torch.float64)     
Float = (torch.float32, torch.float64)
Atom   = (Symbol, Number) 
List   = torch.tensor           
Exp    = (Atom, List)     
Env    = dict             

def standard_env():
	env = Env()
	env.update(vars(torch)) # sin, cos, sqrt, pi, ...
	env.update({
		'+':torch.add, '-':torch.sub, '*':torch.mul, '/':torch.div, 
		'>':torch.greater, '<':torch.less, '>=':torch.greater_equal, '<=':torch.less_equal, '=':torch.equal, 
		'abs':     abs,
		'append':  op.add,  
		'apply':   lambda proc, args: proc(*args),
		'begin':   lambda *x: x[-1],
		'car':     lambda x: x[0],
		'cdr':     lambda x: x[1:], 
		'cons':    lambda x,y: [x] + y,
		'eq?':     op.is_, 
		'expt':    pow,
		'equal?':  op.eq, 
		'length':  len, 
		'list':    lambda *x: List(x), 
		'list?':   lambda x: isinstance(x, List), 
		'map':     map,
		'max':     max,
		'min':     min,
		'not':     op.not_,
		'null?':   lambda x: x == [], 
		'number?': lambda x: isinstance(x, Number),  
		'print':   print,
		'procedure?': callable,
		'round':   round,
		'symbol?': lambda x: isinstance(x, Symbol),
	})
	return env

class Env(dict):
	"An environment: a dict of {'var': val} pairs, with an outer Env."
	def __init__(self, parms=(), args=(), outer=None):
		self.update(zip(parms, args))
		self.outer = outer
	def find(self, var):
		"Find the innermost Env where var appears."
		#print (var)
		return self if (var in self) else self.outer.find(var)

class Procedure(object):
	"A user-defined Scheme procedure."
	def __init__(self, parms, body, env):
		self.parms, self.body, self.env = parms, body, env
	def __call__(self, *args): 
		#print (self.parms, self.body, args)
		#print (Env(self.parms, args, self.env))
		return eval(self.body, Env(self.parms, args, self.env))

global_env = standard_env()

def eval(x, env=global_env):
	"Evaluate an expression in an environment."
	if isinstance(x, Symbol):    # variable reference
		return env.find(x)[x]
	elif not isinstance(x, list):# constant 
		return torch.tensor(x)
	
	op, *args = x       
	
	if op == 'quote':            # quotation
		return args[0]

	elif op == 'if':             # conditional
		#print ('here', args)
		(test, conseq, alt) = args
		#print (test, conseq, alt)
		exp = (conseq if eval(test, env) else alt)
		#print ('there', exp)
		return eval(exp, env)
	elif op == 'define':         # definition
		(symbol, exp) = args
		env[symbol] = eval(exp, env)
	elif op == 'set!':           # assignment
		(symbol, exp) = args
		env.find(symbol)[symbol] = eval(exp, env)
	elif op == 'lambda':         # procedure
		(parms, body) = args
		return Procedure(parms, body, env)
	elif op == 'vector':
		#print ('vec', args)
		out = [eval(i, env) for i in args]
		#return torch.tensor(out)
		try:	
			return torch.stack(out)
		except Exception as e:
			#print ('error', e)
			#print ('here', out)
			return out
	elif op == 'let':
		#print ('\n in let', args)
		env[args[0][0]] = eval(args[0][1], env)
		#print (args[0][0], env[args[0][0]])
		#print (args[1],'\n')
		return eval(args[1], env)
	
	elif op == 'get':
		#print ('here',args)
		#print (args)
		if eval(args[1], env).dtype in ['torch.float32', 'torch.float64']:
			return eval(args[0], env)[float(eval(args[1], env))]
		else:
			return eval(args[0], env)[int(eval(args[1], env))]
	elif op == 'put':
		#print (args)
		vec,a,b = eval(args[0], env), eval(args[1], env), eval(args[2], env)
		#print (vec, a, b)
		if eval(a, env).dtype in ['torch.float32', 'torch.float64']:
			vec[float(a)] = b
		else:
			vec[int(a)] = b
		return (vec)
	elif op == 'first':
		#print ('here', eval(args[0]))
		try:
			return eval(args[0], env)[0]
		except:
			return eval(args[0])[0]
	elif op == 'last':
		return eval(args[0], env)[-1]

	elif op == 'second':
		#print ('here', args)
		try:
			return eval(args[0], env)[1]
		except:
			return eval(args[0])[1]

	elif op =='rest':
		#print ('rest', args)

		return eval(args[0], env)[2:]

	elif op == 'append':
		#print ('in append', args, a, b)
		#print (env)

		#print (eval(args[0]))
		#print (eval(args[1]))
		#print ('\n done append... \n')
		try:
			return torch.cat([eval(args[0], env), eval(args[1], env).unsqueeze(0)])
		except:
			return torch.cat([env[args[0]], env[args[1]].unsqueeze(0)])
	elif op == 'hash-map':
		#print (args)
		hmap = {}
		for i in range(0,len(args),2):
			hmap[args[i]] = eval(args[i+1], env)
		#print (hmap)
		return hmap
	elif op == 'normal':
		#print ('asgas', args)
		return torch.distributions.Normal(eval(args[0], env).float(),eval(args[1], env).float())

	elif op == 'beta':
		return torch.distributions.Beta(eval(args[0], env).float(),eval(args[1], env).float())
	elif op == 'exponential':
		return torch.distributions.Exponential(eval(args[0], env).float())
	elif op == 'uniform':
		return torch.distributions.Uniform(eval(args[0], env).float(),eval(args[1], env).float())
	elif op == 'sample':
		#print ('now sam', args)
		return eval(args[0], env).sample()
	elif op == 'discrete':
		#print ('in dis', args)
		return torch.distributions.Categorical(eval(args[0], env))

	elif op == 'defn':
		#print (args)
		name, param, body = args
		env[name] = Procedure(param, body, env)
		#print ('here')
		return None
		#bre
	elif op == 'observe':
		return eval(args[0], env).sample()
		#return None

	elif op == 'mat-transpose':
		#print (args[0])
		#a = eval(args[0])
		#print (a)
		return torch.transpose(eval(args[0], env),1,0)

	elif op == 'mat-tanh':
		#print (args[0])
		#a = eval(args[0])
		#print (a)
		return torch.tanh(eval(args[0], env))

	elif op == 'mat-add':
		#print ('add',args)
		#print (eval(args[0]).shape, eval(args[1]).shape)
		#a = eval(args[0])
		#print (a)
		return torch.add(eval(args[0], env),eval(args[1], env))
	
	elif op == 'mat-mul':
		#print ('mul',args[0])
		#a = eval(args[0])
		#print (a)
		#bre
		return torch.matmul(eval(args[0], env).float(),eval(args[1], env).float())
	
	elif op == 'mat-repmat':
		#print (args)
		#a = eval(args[0])
		#print (a)
		return eval(args[0], env).repeat(eval(args[1], env).item(),eval(args[2], env).item())
	else:
		#a = env[op]
		#print ('in else',op, args)

		proc = eval(op, env)

		#print ('there',args, proc)

		#print ('see', eval(args[1],env))
		vals = [eval(arg, env) for arg in args]

		#print (vals)

		#return proc(*vals)
		#print ('place')
		return proc(*vals)
	#return None