import math
import numpy as np
from matplotlib import pyplot as plt

def find_factors(n):
    factors = set()
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            factors.add(i)
            factors.add(n // i)

    return sorted(factors)

dataset = "./dataset"
file_ada = dataset + "/Ada/0/0.npy"

img_array = np.load(file_ada)

factors = find_factors(img_array.size)
print(img_array.size)
print(factors)

reshaped_array = img_array.reshape((277, 6))

plt.imshow(reshaped_array, cmap = "gray")
plt.show()