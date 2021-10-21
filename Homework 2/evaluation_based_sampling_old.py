from daphne import daphne
from tests import is_tol, run_prob_test,load_truth
import torch

Symbol = str              # A Scheme Symbol is implemented as a Python str
Number = (torch.int32, torch.float32, torch.float64)     # A Scheme Number is implemented as a Python int or float
Float = (torch.float32, torch.float64)
Atom   = (Symbol, Number) # A Scheme Atom is a Symbol or Number
List   = torch.tensor            # A Scheme List is implemented as a Python list
Exp    = (Atom, List)     # A Scheme expression is an Atom or List
Env    = dict             # A Scheme environment (defined below) 

#import math
import operator as op

def standard_env() -> Env:
    "An environment with some Scheme standard procedures."
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
        return eval(self.body, Env(self.parms, args, self.env))

global_env = standard_env()


def evaluate_program(ast):
    """Evaluate a program as desugared by daphne, generate a sample from the prior
        Args:
            ast: json FOPPL program
        Returns: sample from the prior of ast
    """
    #print ("in eval", len(ast))
    for i in ast:
        val = eval(i)
    return [val], None


def eval(x, env=global_env):
    "Evaluate an expression in an environment."
    if isinstance(x, Symbol):    # variable reference
        #print ('looking for ',x, type(x))
        #if x == 'z':
        #    print ('in z', env.find(x)[x])
        #    return env[x]
        #print ('var ref', x, env.find(x)[x])
        return env.find(x)[x]
    elif not isinstance(x, list):# constant 
        #print ('const ref')
        #return torch.tensor(x, dtype=torch.float32)
        return torch.tensor(x)
    op, *args = x       
    if op == 'quote':            # quotation
        return args[0]
    elif op == 'if':             # conditional
        (test, conseq, alt) = args
        exp = (conseq if eval(test, env) else alt)
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
        out = [eval(i) for i in args]
        #return torch.tensor(out)
        
        try:
           #return torch.tensor([out])
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
        if eval(args[1]).dtype in ['torch.float32', 'torch.float64']:
            return eval(args[0])[float(eval(args[1]))]
        else:
            return eval(args[0])[int(eval(args[1]))]
    elif op == 'put':
        #print (args)
        vec,a,b = eval(args[0]), eval(args[1]), eval(args[2])
        #print (vec, a, b)
        if eval(a).dtype in ['torch.float32', 'torch.float64']:
            vec[float(a)] = b
        else:
            vec[int(a)] = b
        return (vec)
    elif op == 'first':
        return eval(args[0])[0]
    elif op == 'last':
        return eval(args[0])[-1]

    elif op == 'second':
        #print ('here', args)
        return eval(args[0])[1]

    elif op =='rest':
        #print ('rest', args)

        return eval(args[0])[2:]

    elif op == 'append':
        #print ('in append', args, a, b)
        #print (env)

        #print (eval(args[0]))
        #print (eval(args[1]))
        #print ('\n done append... \n')
        try:
            return torch.cat([eval(args[0]), eval(args[1]).unsqueeze(0)])
        except:
            return torch.cat([env[args[0]], env[args[1]].unsqueeze(0)])
    elif op == 'hash-map':
        #print (args)
        hmap = {}
        for i in range(0,len(args),2):
            hmap[args[i]] = eval(args[i+1])
        #print (hmap)
        return hmap
    elif op == 'normal':
        #print ('asgas', args)
        return torch.distributions.Normal(eval(args[0]).float(),eval(args[1]).float())

    elif op == 'beta':
        return torch.distributions.Beta(eval(args[0]).float(),eval(args[1]).float())
    elif op == 'exponential':
        return torch.distributions.Exponential(eval(args[0]).float())
    elif op == 'beta':
        return torch.distributions.Beta(eval(args[0]).float(),eval(args[1]).float())
    elif op == 'uniform':
        return torch.distributions.Uniform(eval(args[0]).float(),eval(args[1]).float())
    elif op == 'sample':
        #print ('now sam', args)
        return eval(args[0]).sample()
    elif op == 'discrete':
        #print ('in dis', args)
        return torch.distributions.Categorical(eval(args[0]))

    elif op == 'defn':
        #print (args)
        name, param, body = args
        env[name] = Procedure(param, body, env)
        return None
        #bre
    elif op == 'observe':
        return None

    elif op == 'mat-transpose':
        #print (args[0])
        #a = eval(args[0])
        #print (a)
        return torch.transpose(eval(args[0]),1,0)

    elif op == 'mat-tanh':
        #print (args[0])
        #a = eval(args[0])
        #print (a)
        return torch.tanh(eval(args[0]))

    elif op == 'mat-add':
        #print ('add',args)
        #print (eval(args[0]).shape, eval(args[1]).shape)
        #a = eval(args[0])
        #print (a)
        return torch.add(eval(args[0]),eval(args[1]))
    
    elif op == 'mat-mul':
        #print ('mul',args[0])
        #a = eval(args[0])
        #print (a)
        #bre
        return torch.matmul(eval(args[0]).float(),eval(args[1]).float())
    
    elif op == 'mat-repmat':
        #print (args)
        #a = eval(args[0])
        #print (a)
        return eval(args[0]).repeat(eval(args[1]).item(),eval(args[2]).item())
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


def get_stream(ast):
    """Return a stream of prior samples"""
    #print (ast)
    while True:
        #for i in ast:
        yield evaluate_program(ast)
    

def run_deterministic_tests():
    
    for i in range(1,14):
        #note: this path should be with respect to the daphne path!
        ast = daphne(['desugar', '-i', '/Users/gaurav/Desktop/CS532-HW2/programs/tests/deterministic/test_{}.daphne'.format(i)])
        
        #print (ast)

        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret, sig = evaluate_program(ast)
        try:
            #print (ret, truth)
            assert(is_tol(ret[0], truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,ast))
        
        print('Test passed')
        
    print('All deterministic tests passed')
    


def run_probabilistic_tests():

    print ('running tests now')
    
    num_samples=100
    max_p_value = 1e-4
    
    for i in range(1,7):
        #note: this path should be with respect to the daphne path!        
        ast = daphne(['desugar', '-i', '/Users/gaurav/Desktop/CS532-HW2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        
        stream = get_stream(ast)
        
        #print ('here \n ', stream, '\n')

        p_val = run_prob_test(stream, truth, num_samples)
        
        print('p value', p_val)
        assert(p_val > max_p_value)
    
    print('All probabilistic tests passed')

        
if __name__ == '__main__':

    #run_deterministic_tests()
    
    run_probabilistic_tests()
    #bre


    for i in range(1,5):
        ast = daphne(['desugar', '-i', '/Users/gaurav/Desktop/CS532-HW2/programs/{}.daphne'.format(i)])

        #print (ast)

        print('\n\n\nSample of prior of program {}:'.format(i))
        print(evaluate_program(ast)[0])