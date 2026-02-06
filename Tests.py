from Mathematics import *
from Methods import *
from Noising import *
from numpy import fft
import time
import os

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

    numpyFM=FrequencyDiscreteFunction(numpyFFT)
    numpyFM=numpyFM.getModule()
    numpyFM.resizeAmplitudeDiscreteFunction()

    FM_discreteImage.show()
    numpyFM.show()

    saveImageFromDiscreteFunction(FM_discreteImage, 'Pictures/DFT_image.png')
    saveImageFromDiscreteFunction(numpyFM, "Pictures/Numpy_image.png")

def RadiusCutTest(path:str, radius:float, x:int=0, y:int=0, centered:bool=False):
    discreteImage=DiscreteFunctionFromImage(path)

    discreteImage.RadiusFilter(radius, x, y, centered)

    showImageFromDiscreteFunction(discreteImage)




def InverseFourierTransformTest(path: str, rayonMax:int=-1):
    discreteImage=DiscreteFunctionFromImage(path)

    h, w = discreteImage.height, discreteImage.width

    start = time.time()
    FM_discreteImage = FourierTransform(discreteImage, rayonMax)
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
    numpyFM.resizeAmplitudeDiscreteFunction()

    Inverse_discreteImage.show()
    numpyFM.show()

    saveImageFromDiscreteFunction(Inverse_discreteImage, 'Pictures/Inverse_image.png')
    saveImageFromDiscreteFunction(numpyFM, "Pictures/Numpy_Inverse_image.png")


def FFT2DTest(path: str):
    im=DiscreteFunctionFromImage(path)

    numpyF=FrequencyDiscreteFunction(fft.fft2(im.kernel))
    numpyFM=numpyF.getModule(True)
    numpyFM.resizeAmplitudeDiscreteFunction()

    myF=FFT2(im)
    myFM=myF.getModule(True)
    myFM.resizeAmplitudeDiscreteFunction()

    numpyFM.show()
    myFM.show()

