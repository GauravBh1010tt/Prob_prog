from daphne import daphne
from tests import is_tol, run_prob_test,load_truth
from primitives import *

import warnings
warnings.filterwarnings("ignore")

env = standard_env()

def evaluate_program(ast, env=global_env):
    """Evaluate a program as desugared by daphne, generate a sample from the prior
        Args:
            ast: json FOPPL program
        Returns: sample from the prior of ast
    """
    #print ("in eval", len(ast))
    #print (global_env)
    #a = global_env['+']
    for i in ast:
        val = eval(i, env)
    return val, env


def get_stream(ast):
    """Return a stream of prior samples"""
    #print (ast)
    while True:
        #for i in ast:
        yield evaluate_program(ast, env)
    

def run_deterministic_tests():
    
    for i in range(1,14):
        #note: this path should be with respect to the daphne path!
        ast = daphne(['desugar', '-i', '/Users/gaurav/Desktop/CS532-HW2/programs/tests/deterministic/test_{}.daphne'.format(i)])

        truth = load_truth('programs/tests/deterministic/test_{}.truth'.format(i))
        ret, sig = evaluate_program(ast)
        
        print ('::Test case::', i, ' ::ast::', ast, '    ::return::', ret)

        #print (ret, truth)
        try:
            assert(is_tol(ret, truth))
        except AssertionError:
            raise AssertionError('return value {} is not equal to truth {} for exp {}'.format(ret,truth,ast))
        
        print('Test passed')
        
    print('All deterministic tests passed')
    


def run_probabilistic_tests():

    print ('running tests now')
    
    num_samples=1e5
    max_p_value = 1e-4
    
    for i in range(1,7):
        #note: this path should be with respect to the daphne path!        
        ast = daphne(['desugar', '-i', '/Users/gaurav/Desktop/CS532-HW2/programs/tests/probabilistic/test_{}.daphne'.format(i)])
        truth = load_truth('programs/tests/probabilistic/test_{}.truth'.format(i))
        
        stream = get_stream(ast)

        p_val = run_prob_test(stream, truth, num_samples)
        
        print ('::Test case::', i, ' ::ast::', ast,' ::p value::', p_val)

        #print('p value', p_val)
        assert(p_val > max_p_value)
    
    print('All probabilistic tests passed')

        
if __name__ == '__main__':

    run_deterministic_tests()
    
    run_probabilistic_tests()
    #bre


    for i in range(1,5):
        ast = daphne(['desugar', '-i', '/Users/gaurav/Desktop/CS532-HW2/programs/{}.daphne'.format(i)])

        #print (ast)

        print('Sample of prior of program {}:'.format(i))
        print(evaluate_program(ast)[0])