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
    
    def parameters(self):
        return self.w + [self.b]
    

# A Layer is a collection of neurons, with a fixed set of input values and output values
# nin is the value passed to individual neuron, nout is the number of neurons that will be created
# calling a layer is summing up all the calls of neuron in a layer
class Layer:
    def __init__(self, nin:int, nout:int):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        # The final layer would only have 1 neuron so return type needs to be malleable
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]  


# A Multi layer perceptron (or an MLP for short) is (as you guessed) a multi layer encapsulation
# it takes in how many inputs (size of x) and then outputs per layer (neurons per layer)
class MLP:
    def __init__(self, nin:int, nouts:list):
        # array of 1 plus array of n
        sizes = [nin] + nouts
        # Each layer needs to connect to the next layer where 0 is nin and the input layer
        # iterating makes output the input iterating from 0 to n-1
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(nouts))]

    def __call__(self, x:list):
        for layer in self.layers:
            # You pass the input to first layer which produces an output
            # which becomes an input to the second layer and so on
            x = layer(x)
        # you return the final input
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]