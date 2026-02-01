import Tests
from numpy import fft

Tests.AdaptativeGaussianFilterTest(r'Pictures\Garden_strawberry.jpg')


exit()
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