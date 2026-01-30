from Mathematics import *
from Methods import *
from Noising import *
import time
from math import cos, pi
from numpy import fft



"""

a = time.time()

f = DiscretefunctionFromImage(r'Garden_strawberry.jpg')
randomNoising(f, -20, 20)

f = (
    DiscretefunctionFromImage(r'Garden_strawberry.jpg')
    .apply(saltAndPaperNoising, 0.01)
    .medianFilter(1)
)

# ou bien

g = DiscretefunctionFromImage(r'Garden_strawberry.jpg')

g.apply(noising, -10, 10)

showImageFromDiscreteFunction(g)




exit()

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
print(f"convolution : {(c - b)/(f.width * f.height)}s per px")"""

#showImageFromDiscreteFunction(g)
#showImageFromDiscreteFunction(f)

#saveDiscreteFunction(f, "jeanClaude.txt")
#showImageFromDiscreteFunction(loadDiscreteFunction("jeanClaude.txt"))

#m=DiscreteFunction([[(k+n)/3 for n in range(128)] for k in range(128)], 0, 0)
#showImageFromDiscreteFunction(m)
#m.resizeAmplitudeDiscreteFunction()
#showImageFromDiscreteFunction(m)

f = DiscretefunctionFromImage(r'rubiks.png')

#Ff=FourierTransform(f, -1)
Ff=FrequencyDiscreteFunction(fft.fft2(f.kernel),0,0)

#saveDiscreteFunction(Ff, "fourier.txt")
Ff_module=Ff.getModule(True)
Ff_module.resizeAmplitudeDiscreteFunction()
showImageFromDiscreteFunction(Ff_module)
saveImageFromDiscreteFunction(Ff_module, "frezFourier.png")

"""
w=1
T=10
r=DiscreteFunction([[127.5*cos(k*w)+127.5 for k in range(int(T*2*pi/w))] for n in range(int(T*2*pi/w))], 0, 0)
#showImageFromDiscreteFunction(r)

Fr=FourierTransform(r)
Fr_module=Fr.getModule(False)
#saveDiscreteFunction(Fr, "cosinus.txt")
Fr_module.resizeAmplitudeDiscreteFunction()
Fr_moduleR=revolveDiscreteFunction(Fr_module)
showImageFromDiscreteFunction(Fr_moduleR)"""