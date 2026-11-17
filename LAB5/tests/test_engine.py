from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from engine import Value


class EngineTests(unittest.TestCase):
    def test_shared_parent(self):
        x = Value(3)
        y = x*x + x
        y.backward()
        self.assertEqual(x.grad,7)
        y.backward()
        self.assertEqual(x.grad,7)

    def test_relu(self):
        for number, expected in [(-2,0),(0,0),(2,1)]:
            x = Value(number)
            x.relu().backward()
            self.assertEqual(x.grad,expected)

    def test_long_graph(self):
        x = Value(2)
        y = x
        for _ in range(2000):
            y = y+1
        y.backward()
        self.assertEqual(x.grad,1)

    def test_zero_power(self):
        x = Value(0)
        (x**0).backward()
        self.assertEqual(x.grad,0)

    def test_multiple_paths(self):
        x = Value(2)
        a = x*x
        (a*a+a).backward()
        self.assertEqual(x.grad,36)


if __name__ == "__main__":
    unittest.main()
