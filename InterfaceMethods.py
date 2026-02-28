from Mathematics import *
from MathematicsMethods import *
from Analysis import *
from PIL import Image

def getInfoForCallback(discreteFunction, infoName):
    a={"Local Variance":LocalVariance, "Gradient Energy":GradientEnergy, "High Frequency Ratio":HighFrequencyRatio, "Historigram":DiscreteFunctionHistorigram}
    return a[infoName](discreteFunction), infoName


def DiscreteFunctionBilateralFilter(discreteFunction:DiscreteFunction, kernelSize:float, sigma_r:float):
    gaussianDiscreteFunction=GaussianDiscreteFunction(kernelSize)

    return discreteFunction.bilateralFilter(gaussianDiscreteFunction, sigma_r)

def DiscreteFunctionAdaptativeFilter(discreteFunction:DiscreteFunction, kernelSize:float, diff:float):
    gaussianDiscreteFunction=GaussianDiscreteFunction(kernelSize)

    return discreteFunction.adaptativeGaussianConvolution(gaussianDiscreteFunction, diff)

def ComplexDiscreteFunctionIFFT2(discreteFunction:ComplexDiscreteFunction):
    ifft2Kernel=IFFT2(discreteFunction.kernel, 2)
    
    IFFT2DiscreteFunction=DiscreteFunction(ifft2Kernel)
    return IFFT2DiscreteFunction

def DiscreteFunctionFFT2(discreteFunction:DiscreteFunction, ) -> DiscreteFunction:
    #if discreteFunction.width*discreteFunction.height > 512**2: fft2Kernel=FFT2Boost(discreteFunction.kernel, pool=mainPool)
    #else: fft2Kernel=FFT2(discreteFunction.kernel)
    fft2Kernel=FFT2(discreteFunction.kernel)

    FFT2DiscreteFunction=ComplexDiscreteFunction(fft2Kernel)

    return FFT2DiscreteFunction

def DiscreteFunctionHistorigram(discreteFunction:DiscreteFunction, width:int=258, height:int=103) -> Image.Image:
    histogram = discreteFunction.getHistorigram()
    maxHisto=max(histogram)

    imL=Image.new("RGBA", (width,height), (0,0,0,0))

    for i in range(width-2):
        x=i/(width-3)*255
        if x==int(x): 
            jMax=histogram[int(x)]
        else: 
            diff=x-int(x)
            jMax=(1-diff)*histogram[int(x)]+diff*histogram[int(x+1)]

        for j in range(int(jMax*(height-3)/maxHisto)):
            xy=(i+1,height-j-1)
            imL.putpixel(xy, (200,100,20,255))
        
    
    imD=imL.copy()

    for im, color in ((imL, (0,0,0,255)), (imD, (255,255,255,255))):
        for i in range(width):
            im.putpixel((i,0), color)
            im.putpixel((i,height-1), color)
        for j in range(height):
            im.putpixel((0,j), color)
            im.putpixel((width-1,j), color)
    
    return imL, imD