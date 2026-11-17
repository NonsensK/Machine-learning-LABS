"""Scalar reverse-mode automatic differentiation."""
import math


class Value:
    def __init__(self, data, parents=(), rule=None):
        self.data = float(data)
        self.grad = 0.0
        self.parents = tuple(parents)
        self.rule = rule or (lambda: None)

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))
        def backward():
            self.grad += out.grad
            other.grad += out.grad
        out.rule = backward
        return out

    __radd__ = __add__

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))
        def backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out.rule = backward
        return out

    __rmul__ = __mul__

    def __pow__(self, power):
        if not isinstance(power, (int, float)):
            raise TypeError("Power must be a scalar constant.")
        out = Value(self.data ** power, (self,))
        def backward():
            if power != 0:
                self.grad += power * self.data ** (power - 1) * out.grad
        out.rule = backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def relu(self):
        out = Value(max(0, self.data), (self,))
        out.rule = lambda: self._accumulate(out.grad * (self.data > 0))
        return out

    def sigmoid(self):
        if self.data >= 0:
            result = 1 / (1 + math.exp(-self.data))
        else:
            exp = math.exp(self.data)
            result = exp / (1 + exp)
        out = Value(result, (self,))
        out.rule = lambda: self._accumulate(out.grad * result * (1 - result))
        return out

    def log(self):
        out = Value(math.log(self.data), (self,))
        out.rule = lambda: self._accumulate(out.grad / self.data)
        return out

    def _accumulate(self, gradient):
        self.grad += gradient

    def backward(self):
        # Iterative DFS supports long sums without Python recursion limits.
        order, visited = [], set()
        stack = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
            elif node not in visited:
                visited.add(node)
                stack.append((node, True))
                stack.extend((parent, False) for parent in reversed(node.parents))
        # One backward call computes fresh gradients. Contributions within it accumulate.
        for node in order:
            node.grad = 0.0
        self.grad = 1.0
        for node in reversed(order):
            node.rule()
