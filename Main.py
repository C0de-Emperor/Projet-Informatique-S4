from Mathematics import DiscretefunctionFromImage, DiscreteFunction
from Methods import *
import time

a = time.time()

f = DiscretefunctionFromImage(r'Garden_strawberry.jpg')

b = time.time()
print(f"convertion : {b - a}s")
print(f"convertion : {(b - a)/(f.width * f.height)}s per px")


h = DiscreteFunction(
    [
        [1, 2, 1],
        [2, 3, 2],
        [1, 2, 1]
    ],
    x = 0,
    y = 0
)

f.convolve(h)

c = time.time()

print(f"convolution : {c - b}s")
print(f"convolution : {(c - b)/(f.width * f.height)}s per px")

'''
print(getNiveauxGris(input("chemin de l'image ? : "), (0.299, 0.587, 0.114), True))

f = DiscreteFunction(
    [
        [255, 40, 30, 20, 10],
        [39, 38, 138, 130, 12],
        [7, 210, 186, 1, 1],
        [200, 210, 186, 1, 1],
        [100, 210, 186, 1, 1]
    ],
    x = 0,
    y = 0
)



print(f.convolve(h).kernel)
'''