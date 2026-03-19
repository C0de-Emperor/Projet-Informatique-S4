from Mathematics import *
from ImageMethods import *
from Noising import *
from Analysis import *
from numpy import fft, linspace
from matplotlib import pyplot as plt
from math import ceil
import time
import os

# Filters

def MedianFilterTest (path: str):
    if not os.path.exists(path):
        raise FileExistsError(f"{path} does not exist")

    a = time.time()

    f = (
        DiscreteFunctionFromImage(path)
        .apply(SaltAndPaperNoising, 0.07)
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
        .apply(RandomNoising, -30, 30)
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
        .apply(RandomNoising, -30, 30)
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

# Fourier

def InverseFourierTransformTest(path: str, rayonMax:int=-1):
    discreteImage=DiscreteFunctionFromImage(path)

    h, w = discreteImage.height, discreteImage.width

    start = time.time()
    FM_discreteImage = FourierTransform(discreteImage)
    print("DFT :", time.time() - start)

    matrice: list[list] = [] # car sinon probleme de fréquence et informations en trop
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
            pixel = Inverse_discreteImage[w - 1 - i, h - 1 - j]
            ligne.append(pixel)
        matrice_finale.append(ligne)

    Inverse_discreteImage = ComplexDiscreteFunction(matrice_finale)

    Inverse_discreteImage = Inverse_discreteImage.getModule(False)
    Inverse_discreteImage.resizeAmplitudeDiscreteFunction()

    start = time.time()
    numpyFFT = fft.fft2(discreteImage.kernel)
    print("Numpy FFT :", time.time() - start)

    start = time.time()
    numpyInverse = fft.ifft2(numpyFFT)
    print("Numpy Inverse FFT :", time.time() - start)

    numpyFM = ComplexDiscreteFunction(numpyInverse.tolist())
    numpyFM = numpyFM.getModule(False)
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

    RandomNoising(im, int(noiseIntensity*10), int(noiseIntensity*100))
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
    fourierTransform=[]
    dft=[]

    for n in range(start, end, step):
        pixels.append(n)
        a=[[i*j for i in range(n)] for j in range(n)]

        startTime=time.time()
        #FourierTransform(a)
        fourierTransform.append(time.time()-startTime)

        startTime=time.time()
        #DFT(a)
        dft.append(time.time()-startTime)

        startTime=time.time()
        fft.fft2(a)
        numpySeries.append(time.time()-startTime)

        startTime=time.time()
        FFT2(a, 2)
        homemadeSeries.append(time.time()-startTime)

        #if __name__=="__main__":
        startTime=time.time()
        FFT2Boost(a, 2)
        boostedHomemadeSeries.append(time.time()-startTime)

        print("--------", int((n-start+step)*100/(end-start)), "%")
    
    plt.plot(pixels, numpySeries, "-b", label="fft numpy")
    plt.plot(pixels, homemadeSeries, "-r", label="notre fft")
    plt.plot(pixels, boostedHomemadeSeries, "-g", label="notre fft+multiprocessing")
    plt.plot(pixels, fourierTransform, "-m", label="transformée de Fourier classique")
    plt.plot(pixels, dft, "-c", label="dft sur lignes puis colonnes")
    plt.legend(loc="upper left")
    plt.xlabel("pixels on image's side")
    plt.ylabel("time taken to compute the FFT")

    plt.show()

def FFTAmplitudeCutTest(path:str, maxAmp:float, isLog:bool=False):
    im=DiscreteFunctionFromImage(path)
    SaltAndPaperNoising(im, 0.05)
    im.show()

    imF=ComplexDiscreteFunction(FFT2(im.kernel))
    imF.maxAmplitudeCut(maxAmp, isLog)

    im2=DiscreteFunction(IFFT2(imF.kernel, 2))
    im2.show()

def SectionnedMultiprocessedFFT2Test(imageSize, start, end, step):
    sections=[]
    sectionnedBoostedFFT2=[]

    a=[[i*j for i in range(imageSize)] for j in range(imageSize)]

    for n in range(start, end, step):
        sections.append(n)

        #if __name__=="__main__":
        startTime=time.time()
        sectionnedFFT2Boost(a, n)
        sectionnedBoostedFFT2.append(time.time()-startTime)

        print("--------", int((n-start+step)*100/(end-start)), "%")
    
    #if __name__=="__main__":
    startTime=time.time()
    FFT2Boost(a, 2)
    boostedFFT2=[time.time()-startTime]*len(sections)

    plt.plot(sections, boostedFFT2, "-g", label="controle (fft2 avec multiprocessing)")
    plt.plot(sections, sectionnedBoostedFFT2, "-c", label="fft2 avec multiprocessing et segmentation")
    plt.legend(loc="upper left")
    plt.xlabel("number of sections for multiprocessing")
    plt.ylabel("time taken to compute the FFT")

    plt.show()

def TrueConvolutionTest(discreteFunction:DiscreteFunction, kernel:DiscreteFunction, convoTest=False):
    a=discreteFunction
    b=kernel

    #w=10
    #b=DiscreteFunction([[(abs(i-(a.height-j))<=1)*((a.width//2-w<i<a.width//2+w)*(a.height//-w<j<a.height//2+w)) for i in range(a.width)] for j in range(a.height)])
    

    if convoTest:
        startTime=time.time()
        d=a.convolve(b)
        d.show()
        print("convolution:", time.time()-startTime)

    startTime=time.time()
    (a2, coordinates)=a.extend((2**ceil(log(a.width,2)), 2**ceil(log(a.height,2))))

    (b2,raf)=b.extend((a2.width, a2.height))
    b2=b2.getCentered()
    #b2.normalize()

    af=ComplexDiscreteFunction(FFT2(a2.kernel))
    b2f=ComplexDiscreteFunction(FFT2(b2.kernel))

    cf=af*b2f

    c=DiscreteFunction(IFFT2(cf.kernel))
    c.resize(coordinates=coordinates)
    c.show()
    print("frequency multiplication:", time.time()-startTime)

    ef=cf/b2f
    e=DiscreteFunction(IFFT2(ef.kernel))
    e.resize(coordinates=coordinates)
    e.show()


def test_couleur(path):
    image = ColorDiscreteFunctionFromImage(path)
    image.show()

    f = GaussianDiscreteFunction(5, 1)
    d = 30.0
    #image.apply_to_all("adaptativeGaussianConvolution", f, d)
    image.apply_to_all("medianFilter",3)

    image.show()

