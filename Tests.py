from Mathematics import *
from ImageMethods import *
from Noising import *
from Analysis import *
from numpy import fft
from matplotlib import pyplot as plt
import random
import time
import os

def AnalysisSaltAndPaperTest (path: str, probability: float):
    if not os.path.exists(path):
        raise FileExistsError(f"{path} does not exist")
    
    f = DiscreteFunctionFromImage(path)
    print("Local variance before : ", LocalVariance(f))
    print("Gradient Energy before : ", GradientEnergy(f))
    print("Quality Score before : ", QualityScore(f))
    print("")

    f.apply(saltAndPaperNoising, probability)

    print("Local variance with noise : ", LocalVariance(f))
    print("Gradient Energy with noise : ", GradientEnergy(f))
    print("Quality Score with noise : ", QualityScore(f))
    print("")

    f.medianFilter(1)

    print("Local variance after : ", LocalVariance(f))
    print("Gradient Energy after : ", GradientEnergy(f))
    print("Quality Score after : ", QualityScore(f))

def AnalysisAdaptativeGaussianTest (path: str, minV: int, maxV: int, sigma:float, diff: float):
    if not os.path.exists(path):
        raise FileExistsError(f"{path} does not exist")
    
    f = DiscreteFunctionFromImage(path)
    print("Local variance before : ", LocalVariance(f))
    print("Gradient Energy before : ", GradientEnergy(f))
    print("Quality Score before : ", QualityScore(f))
    print("")

    f.apply(randomNoising, minV, maxV)

    f.show()

    print("Local variance with noise : ", LocalVariance(f))
    print("Gradient Energy with noise : ", GradientEnergy(f))
    print("Quality Score with noise : ", QualityScore(f))
    print("")

    h = GaussianDiscreteFunction(sigma)
    g = f.adaptativeGaussianConvolution(h, diff)

    print("Local variance after : ", LocalVariance(g))
    print("Gradient Energy after : ", GradientEnergy(g))
    print("Quality Score after : ", QualityScore(g))

    g.show()

def MedianFilterTest (path: str):
    if not os.path.exists(path):
        raise FileExistsError(f"{path} does not exist")

    a = time.time()

    f = (
        DiscreteFunctionFromImage(path)
        .apply(saltAndPaperNoising, 0.07)
    )
    f.show()

    b = time.time()
    print(f"convertion : {b - a}s")
    print(f"convertion : {(b - a)/(f.width * f.height)}s per px", end="\n"*2)

    f.medianFilter(1)

    c = time.time()

    print(f"median filter : {c - b}s")
    print(f"median filter : {(c - b)/(f.width * f.height)}s per px")

    f.show()

def GaussianFilterTest (path: str):
    if not os.path.exists(path):
        raise FileExistsError(f"{path} does not exist")

    a = time.time()

    f = (
        DiscreteFunctionFromImage(path)
        .apply(randomNoising, -30, 30)
    )
    f.show()

    h = GaussianDiscreteFunction(0.6)

    b = time.time()
    print(f"convertion : {b - a}s")
    print(f"convertion : {(b - a)/(f.width * f.height)}s per px", end="\n"*2)

    g = f.convolve(h)

    c = time.time()

    print(f"gaussian filter : {c - b}s")
    print(f"gaussian filter : {(c - b)/(f.width * f.height)}s per px")

    g.show()

def AdaptativeGaussianFilterTest (path: str):
    if not os.path.exists(path):
        raise FileExistsError(f"{path} does not exist")

    a = time.time()

    f = (
        DiscreteFunctionFromImage(path)
        .apply(randomNoising, -30, 30)
    )
    f.show()

    h = GaussianDiscreteFunction(0.6)

    b = time.time()
    print(f"convertion : {b - a}s")
    print(f"convertion : {(b - a)/(f.width * f.height)}s per px", end="\n"*2)

    g = f.adaptativeGaussianConvolution(h, 50)

    c = time.time()

    print(f"adaptative gaussian filter : {c - b}s")
    print(f"adaptative gaussian filter : {(c - b)/(f.width * f.height)}s per px")

    g.show()

def FourierTransformTest(path: str, rayonMax:int=-1):
    discreteImage=DiscreteFunctionFromImage(path)

    start=time.time()
    FM_discreteImage=FourierTransform(discreteImage, rayonMax)
    print("DFT :", time.time()-start)

    FM_discreteImage=FM_discreteImage.getModule()
    FM_discreteImage.resizeAmplitudeDiscreteFunction()

    start=time.time()
    numpyFFT=fft.fft2(discreteImage.kernel)
    print("Numpy FFT :", time.time()-start)

    numpyFM=ComplexDiscreteFunction(numpyFFT)
    numpyFM=numpyFM.getModule()
    numpyFM.resizeAmplitudeDiscreteFunction()

    FM_discreteImage.show()
    numpyFM.show()

    saveImageFromDiscreteFunction(FM_discreteImage, 'Pictures/DFT_image.png')
    saveImageFromDiscreteFunction(numpyFM, "Pictures/Numpy_image.png")

def RadiusCutTest(path:str, radius:float, x:int=0, y:int=0, centered:bool=False):
    discreteImage=DiscreteFunctionFromImage(path)

    discreteImage.RadiusFilter(radius, x, y, centered)

    discreteImage.show()

def InverseFourierTransformTest(path: str, rayonMax:int=-1):
    discreteImage=DiscreteFunctionFromImage(path)

    h, w = discreteImage.height, discreteImage.width

    start = time.time()
    FM_discreteImage = FourierTransform(discreteImage)
    print("DFT :", time.time() - start)

    matrice = [] # car sinon probleme de fréquence et informations en trop
    for j in range(h):
        matrice.append([])
        for i in range(w):
            matrice[j].append(FM_discreteImage.kernel[j][i])

    clean_discreteImage = DiscreteFunction(matrice)

    start = time.time()
    Inverse_discreteImage = InverseFourierTransform(clean_discreteImage, rayonMax)
    print("IDFT :", time.time() - start)

    matrice_finale = [] # on retourne l'image
    for j in range(h):
        ligne = []
        for i in range(w):
            pixel = Inverse_discreteImage.kernel[h - 1 - j][w - 1 - i]
            ligne.append(pixel)
        matrice_finale.append(ligne)

    Inverse_discreteImage = DiscreteFunction(matrice_finale)

    Inverse_discreteImage=Inverse_discreteImage.getModule(False)
    Inverse_discreteImage.resizeAmplitudeDiscreteFunction()

    start = time.time()
    numpyFFT = fft.fft2(discreteImage.kernel)
    print("Numpy FFT :", time.time() - start)

    start = time.time()
    numpyInverse = fft.ifft2(numpyFFT)
    print("Numpy Inverse FFT :", time.time() - start)

    numpyFM=DiscreteFunction(numpyInverse.tolist())
    numpyFM=numpyFM.getModule(False)
    numpyFM.resizeAmplitude()

    Inverse_discreteImage.show()
    numpyFM.show()

    saveImageFromDiscreteFunction(Inverse_discreteImage, 'Pictures/Inverse_image.png')
    saveImageFromDiscreteFunction(numpyFM, "Pictures/Numpy_Inverse_image.png")


def FFT2DTest(path: str, completionMode:int):
    im=DiscreteFunctionFromImage(path)

    startTime=time.time()

    """numpyF=ComplexDiscreteFunction(fft.fft2(im.kernel))
    print(time.time()-startTime)
    numpyFM=numpyF.getModule(True)
    numpyFM.resizeAmplitude()
    numpyFMR=numpyFM.getRevolve()

    numpyFM.show()"""

    startTime=time.time()

    myF=ComplexDiscreteFunction(FFT2(im.kernel, completionMode))
    print(time.time()-startTime)
    myFM=myF.getModule(True)
    myFM.resizeAmplitude()
    myFMR=myFM.getRevolve()

    myFM.show()

def IFFT2DTest(path:str, completionMode:int):
    im=DiscreteFunctionFromImage(path)

    myF=FFT2(im.kernel, completionMode)
    
    newIm=ComplexDiscreteFunction(IFFT2(myF, completionMode))
    newIm=newIm.getModule()
    newIm.resizeAmplitude()

    newIm.show()

def FFTRadiusCutTest(path:str, noiseIntensity:float, radius:float):
    im=DiscreteFunctionFromImage(path)

    randomNoising(im, int(noiseIntensity*10), int(noiseIntensity*100))
    im.show()

    a=FFT2(im.kernel, 2)
    imF=ComplexDiscreteFunction(a)

    imF.RadiusFilter(radius)

    im2=DiscreteFunction(IFFT2(imF.kernel, 2))
    im2.show()

def FTsTimeTest(start, end, step):
    pixels=[]
    numpySeries=[]
    homemadeSeries=[]
    boostedHomemadeSeries=[]
    dft=[]

    for n in range(start, end, step):
        pixels.append(n)
        a=[[i*j for i in range(n)] for j in range(n)]

        startTime=time.time()
        fft.fft2(a)
        numpySeries.append(time.time()-startTime)

        startTime=time.time()
        FFT2(a, 2)
        homemadeSeries.append(time.time()-startTime)

        #if __name__=="__main__":
        startTime=time.time()
        #FFT2Boost(a, 2)
        boostedHomemadeSeries.append(time.time()-startTime)

        startTime=time.time()
        FourierTransform(a)
        dft.append(time.time()-startTime)

        print(int((n-start+step)*100/(end-start)), "%")
    
    plt.plot(pixels, numpySeries, "-b", label="fft numpy")
    plt.plot(pixels, homemadeSeries, "-r", label="notre fft")
    plt.plot(pixels, boostedHomemadeSeries, "-g", label="notre fft+multiprocessing")
    plt.plot(pixels, dft, "-m", label="notre dft classique")
    plt.legend(loc="upper left")
    plt.xlabel("pixels on image's side")
    plt.ylabel("time taken to compute the FFT")

    plt.show()

def FFTAmplitudeCutTest(path:str, maxAmp:float, isLog:bool=False):
    im=DiscreteFunctionFromImage(path)
    saltAndPaperNoising(im, 0.05)
    im.show()

    imF=ComplexDiscreteFunction(FFT2(im.kernel))
    imF.maxAmplitudeCut(maxAmp, isLog)

    im2=DiscreteFunction(IFFT2(imF.kernel, 2))
    im2.show()