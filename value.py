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
        return f"Value(data={self.data} label={self.label} grad={self.grad})"
    
    # Adding to values and applying chain rule to gradient for backward propogation
    # Rule applied: Addition transfers gradient from the output gradient directly
    def __add__(self, other):
        # Assuming if not Value then int
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out; 
    
    ## to solve 3 + obj of value
    def __radd__(self, other):
        return self + other

    # Multiplying the values and applying chain rule to gradient for backward propogation
    # Rule applied: Multiply uses chain rule by multipling the multipliers value to outs grad
    def __mul__(self, other):
         # Assuming if not Value then int
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        return self * other**-1
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        
        return out
    
    def __pow__(self, k):
        assert isinstance(k, (int , float)), "supports only int / float" 
        out = Value(self.data**k, (self, ), f'**{k}')

        def _backward():
            # derivative of dx**a/dx = ax**(a-1)
            self.grad += k * (self.data ** (k-1)) * out.grad
        out._backward = _backward
        return out;
    
    
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
        