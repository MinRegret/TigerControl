
import jax
import jax.numpy as np

#f = lambda x: np.abs(x)
def f(x):
	return np.abs(x)

def g(x, func):
	return func(func(x))


print("test")
h = jax.jit(g)

print("output")
g(-1, f)
h(-1, f)