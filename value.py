class Value():

    # Constructor
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.grad = 1.0

    # str representation of the object
    # str('hello') returns 'hello', while repr('hello') returns "'hello'" (with quotes inside the string) to show it's a string literal
    def __repr__(self):
        return f"Value(data={self.data} label={self.label})"
    
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        return out; 
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        self.grad = other.data
        other.grad = self.data
        return out
