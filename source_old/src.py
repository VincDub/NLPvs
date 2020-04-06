# %%
from numba import jit
import math
import timeit
import jupyter

@jit
def hypot(x, y):
    x = abs(x)
    y = abs(y)
    t = min(x, y)
    x = max(x, y)
    t = t / x
    return x * math.sqrt(1+t*t)

print(hypot(3,4))
print(hypot.py_func(3.0, 4.0))

# %%
# Comparaison des performances grâce à Interactive Python
%timeit hypot.py_func(3.0, 4.0)
%timeit hypot(3.0, 4.0)

# %%
