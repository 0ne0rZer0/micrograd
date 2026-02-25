import math
class Value():

    # Constructor
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.grad = 0.0
        self._backward = lambda:None

    # str representation of the object
    # str('hello') returns 'hello', while repr('hello') returns "'hello'" (with quotes inside the string) to show it's a string literal
    def __repr__(self):
        return f"Value(data={self.data} label={self.label})"
    
    # Adding to values and applying chain rule to gradient for backward propogation
    # Rule applied: Addition transfers gradient from the output gradient directly
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out; 
    
    # Multiplying the values and applying chain rule to gradient for backward propogation
    # Rule applied: Multiply uses chain rule by multipling the multipliers value to outs grad
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out
    
    # Tanh is a direct forumla so is it's derivative (1 - tanh^2(n))
    def tanh(self):
        x = self.data
        t = (math.exp(2*x)-1) / (math.exp(2*x)+1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            # inside the close self is n, and out is o when o = n.tanh()
            # so this translates to
            # n.grad += 1-tanh^2(n) * o.grad (1 here)
            # += is done because gradiantes accumulate. 
            # a = b + c and d = b*e b is influencing both.
            # Multiplication happens along a single path in chain rule
            # Addition happens when multiple paths merge in chain rule
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
        