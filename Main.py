from Mathematics import *
from Methods import *
from Noising import *
import time





a = time.time()

f = DiscretefunctionFromImage(r'Garden_strawberry.jpg')
noising(f, -20, 20)

showImageFromDiscreteFunction(f)



h = GaussianDiscreteFunction(0.6)

b = time.time()
print(f"convertion : {b - a}s")
print(f"convertion : {(b - a)/(f.width * f.height)}s per px")

"""
h = DiscreteFunction(
    [
        [0, 1, 0],
        [1, 0, -1],
        [0, -1, 0]
    ],
    x = 0,
    y = 0
)

h.normalize()
"""

g = f.convolve(h)

c = time.time()

print(f"convolution : {c - b}s")
print(f"convolution : {(c - b)/(f.width * f.height)}s per px")

#showImageFromDiscreteFunction(g)
#showImageFromDiscreteFunction(f)

#saveDiscreteFunction(f, "jeanClaude.txt")
#showImageFromDiscreteFunction(loadDiscreteFunction("jeanClaude.txt"))

#m=DiscreteFunction([[(k+n)/3 for n in range(128)] for k in range(128)], 0, 0)
#showImageFromDiscreteFunction(m)
#m.resizeAmplitudeDiscreteFunction()
#showImageFromDiscreteFunction(m)

Ff=FourierTransform(f, 0)
saveDiscreteFunction(Ff, "fourier.txt")
Ff_module=Ff.getModule(True)
Ff_module.resizeAmplitudeDiscreteFunction()
showImageFromDiscreteFunction(Ff_module)

