---
lastSync: Fri Feb 27 2026 00:15:24 GMT+0530 (India Standard Time)
tags:
  - ai_ml
  - micrograd
  - karpathy
  - backpropagation
  - neural_networks
  - phase1
---
Reference: [Karpathy micrograd video](https://www.youtube.com/watch?v=VMj-3S1tku0)
Code: git@github.com:0ne0rZer0/micrograd.git

### What is micrograd?

An autograd engine from scratch. Tracks math operations, builds a graph, walks backward computing gradients. GPT uses the same idea, just bigger.

---

### How do derivatives work visually?

Take $f(x) = x^2$. Plot it:

```
    f(x)
    |         *
    |       *
    |     *
    |   *
    |  *
    | *
    |*
    *------------- x
```

The derivative at any point = the slope of the curve at that point

```
    f(x)
    |         *
    |       * /  <-- steep here = large derivative
    |     * /
    |   */
    |  */
    | */
    |*--- <-- flat here = small derivative
    *------------- x
```

The formula says the same thing with numbers:

$$
\frac{df}{dx} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
$$

Pick a point $x$. Nudge it by tiny $h$. Measure how much $f$ changed. Divide by $h$. That ratio = the slope.

Concrete example, $f(x) = x^2$ at $x = 3$:
- $f(3) = 9$
- $f(3.001) = 9.006001$
- Slope $\approx \frac{9.006 - 9}{0.001} = 6.0$
- Derivative of $x^2$ is $2x$, and $2 \times 3 = 6$. Checks out.

**Why this matters for neural nets:** every parameter has a derivative with respect to the loss. That derivative tells you which direction to move the parameter to bring the loss down.

---

### What does the Value class track?

- `data` = the actual number
- `grad` = how much loss changes if this value is nudged (starts at 0)
- `_prev` = which Values produced this one (the children)
- `_op` = what operation created it
- `_backward` = function to compute gradients for this op

Every operation (`+`, `*`, `**`, `tanh`, `exp`) creates a new Value and stores how to send gradients backward.

---

### What is the computational graph?

`e = a*b`, then `d = e + c`, then `L = f*d` builds:

```
a --\
     (*)--> e --\
b --/            (+)--> d --\
            c --/            (*)--> L
                        f --/
```

Left to right = forward pass (computing output)
Right to left = backward pass (computing gradients)

---

### How does backpropagation flow through the graph?

For every value $v$: what is $\frac{\partial L}{\partial v}$?

Start at the output. $\frac{\partial L}{\partial L} = 1$ (nudging $L$ by 1 changes $L$ by 1).

Walk backward through each operation:

```
                                    grad
a --\                               a: ?
     (*)--> e --\                   e: ?
b --/            (+)--> d --\       d: ?
            c --/            (*)--> L: 1.0  <-- start here
                        f --/       f: ?
                                    c: ?
                                    b: ?
```

Each operation has a local rule for passing gradients back. Multiply the incoming gradient by the local derivative. That is the chain rule.

#### The chain rule visually

```
  e ----[some op]----> d ----[some op]----> L

  dL/de  =  dL/dd  *  dd/de
             ^^^^      ^^^^
             "how L     "how d
             reacts     reacts
             to d"      to e"
```

Multiply along the path. Multiple paths from $e$ to $L$? Add them up.

---

### What are the gradient rules per operation?

| Operation | How gradients flow back | Intuition |
|-----------|------------------------|-----------|
| $c = a + b$ | `a.grad += 1.0 * out.grad` | Addition passes gradient through |
| | `b.grad += 1.0 * out.grad` | Both inputs get the same gradient |
| $c = a \times b$ | `a.grad += b.data * out.grad` | Swap the values |
| | `b.grad += a.data * out.grad` | Multiply by the other input |
| $c = a^k$ | `a.grad += k * a**(k-1) * out.grad` | Power rule |
| $o = \tanh(n)$ | `n.grad += (1 - t**2) * out.grad` | `t = tanh(n)` |
| $o = e^x$ | `x.grad += out.data * out.grad` | $e^x$ derivative is itself |

#### Worked example: multiplication

```
a = 2, b = -3, c = a*b = -6
out.grad = 4.0 (from downstream)

a.grad += b.data * out.grad = (-3) * 4.0 = -12.0
b.grad += a.data * out.grad = (2) * 4.0  = 8.0

Check: a.grad is negative.
Increasing a (positive) times b (negative) makes c more negative.
That pushes output down. Negative gradient = "go down". Correct.
```

#### Worked example: addition

```
e = -6, c = 10, d = e + c = 4
out.grad = -2.0 (from downstream)

e.grad += 1.0 * (-2.0) = -2.0
c.grad += 1.0 * (-2.0) = -2.0

Both get the same gradient. Addition splits the signal equally.
```

---

### Why `+=` and not `=` for gradients?

If a value feeds into two operations:

```
       /--> [op1] --> ...
  b --<
       \--> [op2] --> ...
```

Gradient from both paths adds up.

```python
# += is done because gradients accumulate.
# a = b + c and d = b*e, b is influencing both.
# Multiplication happens along a single path
# Addition happens when multiple paths merge
```

Using `=` would overwrite the first path's contribution. Wrong.

---

### Why topological sort?

Gradients flow backward. Every node needs to be done after all nodes it feeds into.

```
Forward:   a, b --> e --> d --> L
Backward:  L --> d --> e --> a, b
```

DFS that appends after visiting children, then reverse:

```python
def build_topo(v):
    if v not in visited:
        visited.add(v)
        for child in v._prev:
            build_topo(child)
        topo.append(v)
```

Set `L.grad = 1.0`, walk backward calling `_backward()` on each.

---

### How does a Neuron work?

```
x1 --(*w1)-\
            \
x2 --(*w2)---(sum + b)---[tanh]---> output
            /
x3 --(*w3)-/
```

$$
\text{out} = \tanh\left(\sum_{i} w_i \cdot x_i + b\right)
$$

- Weights $w_i$: random at start, these are what get learned
- Bias $b$: random, also learned
- `tanh` squishes output to $[-1, 1]$
- Parameters per neuron: `nin` weights + 1 bias

---

### How do Layer and MLP build on Neuron?

**Layer** = list of neurons, same inputs

```
          [Neuron 0] --> out0
x1, x2 --[Neuron 1] --> out1
          [Neuron 2] --> out2
```

`Layer(2, 3)` = 3 neurons each taking 2 inputs

**MLP** = layers chained, output of one feeds the next

```
         Layer 0          Layer 1          Layer 2
x1 \   [N][N][N][N]    [N][N][N][N]       [N]
x2  >-->  4 neurons  -->  4 neurons  -->  1 neuron --> output
x3 /
```

`MLP(3, [4, 4, 1])`: sizes = `[3, 4, 4, 1]`
- Layer 0: 3 in, 4 out
- Layer 1: 4 in, 4 out
- Layer 2: 4 in, 1 out

Parameters: $(3 \times 4 + 4) + (4 \times 4 + 4) + (4 \times 1 + 1) = 41$

---

### What is the loss function?

$$
L = \sum_{i} (y_{\text{out}}^{(i)} - y_{\text{ideal}}^{(i)})^2
$$

```
ideal:  [ 1.0, -1.0, -1.0,  1.0]
actual: [ 0.9, -0.3,  0.6,  0.8]
errors:  0.01   0.49   2.56   0.04
loss = 3.1
```

Squaring makes bigger errors hurt more. Goal: loss close to 0.

---

### How does gradient descent work?

$$
p_{\text{new}} = p_{\text{old}} - \alpha \cdot \frac{\partial L}{\partial p}
$$

Visually:

```
loss
  |   *
  |  * *        <-- too high learning rate, bouncing
  | *   *
  |*     *
  *-------*---- parameter value

loss
  |   *
  |    *
  |     *
  |      *      <-- good learning rate, sliding down
  |       *
  *------------- parameter value
```

```
grad is positive:   increasing p increases loss
                    subtract --> p goes down --> loss goes down

grad is negative:   increasing p decreases loss
                    subtracting a negative = adding --> p goes up --> loss goes down
```

$\alpha$ = learning rate. 0.01 to 0.05 worked for this net.

---

### The full training loop

```python
for i in range(n):
    # zero grads (they accumulate, must reset)
    for p in n.parameters():
        p.grad = 0.0

    # forward
    y_outputs = [n(x) for x in x_inputs]
    loss = sum((y_o - y_ideal)**2 for y_o, y_ideal in zip(y_ideals, y_outputs))

    # backward
    loss.backward()

    # update
    for p in n.parameters():
        p.data += -0.01 * p.grad
```

```
Zero --> Forward --> Loss --> Backward --> Update
  ^                                          |
  |__________________________________________|
                    repeat
```

---

### Things to remember

- **Zero gradients before each pass.** They accumulate with `+=`. Forget this and loss goes haywire.
- **tanh = raw exp math.** $\tanh(x) = \frac{e^{2x}-1}{e^{2x}+1}$. Same gradients. Graph shape doesn't matter, math does.
- **Learning rate.** 0.01 to 0.05 for this net. Too high = overshoot. Too low = slow.
- **PyTorch does exactly this.** Same graph, same backward. Optimized in C++ with GPU.

---