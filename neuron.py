from value import Value
import random
# A neuron is a collection of math expression
# Here it is a summation of wi*xi i ranges to a given input nin, + activation bias b

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))
    
    def __call__(self, x:list):
        # Call is making this class as a callable func
        # n = Neuron(4); n(10) calls this
        # sum function zips up x[0]-x[nin] with a w[0] to w[nin] and sums the mult of all wi*xi
        # second parameter is just starting value of sum \
        # which will be bias to complete the neuron forumla of sum(wi*xi) + b for all i
        value = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)

        # activation fucntion
        out = value.tanh()
        return out