from Mathematics import *
from ImageMethods import *
from Noising import *
from Analysis import *
from numpy import fft, linspace
from matplotlib import pyplot as plt
import time
import os

# Analysis

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

# Analysis Curves

import os
import matplotlib.pyplot as plt
import numpy as np

def AnalysisSaltAndPaperCurveVSMedian(path: str, p_max=0.5, steps=20):

    if not os.path.exists(path):
        raise FileExistsError(f"{path} does not exist")

    probs = np.linspace(0, p_max, steps)

    # Valeurs stockées
    var_noise = []
    var_filtered = []

    grad_noise = []
    grad_filtered = []

    score_noise = []
    score_filtered = []

    # Image originale (une seule fois)
    f_original = DiscreteFunctionFromImage(path)

    var_original = LocalVariance(f_original)
    grad_original = GradientEnergy(f_original)
    score_original = QualityScore(f_original)

    for p in probs:
        f = DiscreteFunctionFromImage(path)

        # Bruitage
        f.apply(saltAndPaperNoising, p)

        var_noise.append(LocalVariance(f))
        grad_noise.append(GradientEnergy(f))
        score_noise.append(QualityScore(f))

        # Filtrage
        f.medianFilter(1)

        var_filtered.append(LocalVariance(f))
        grad_filtered.append(GradientEnergy(f))
        score_filtered.append(QualityScore(f))

    # =========================
    # 1 - Variance
    # =========================
    plt.figure()
    plt.plot(probs, [var_original]*len(probs))
    plt.plot(probs, var_noise)
    plt.plot(probs, var_filtered)
    plt.xlabel("Probability")
    plt.ylabel("Local Variance")
    plt.title("Local Variance vs Salt & Pepper Probability")
    plt.legend(["Original", "With Noise", "After Median"])
    plt.show()

    # =========================
    # 2 - Gradient
    # =========================
    plt.figure()
    plt.plot(probs, [grad_original]*len(probs))
    plt.plot(probs, grad_noise)
    plt.plot(probs, grad_filtered)
    plt.xlabel("Probability")
    plt.ylabel("Gradient Energy")
    plt.title("Gradient Energy vs Salt & Pepper Probability")
    plt.legend(["Original", "With Noise", "After Median"])
    plt.show()

    # =========================
    # 3 - Quality Score
    # =========================
    plt.figure()
    plt.plot(probs, [score_original]*len(probs))
    plt.plot(probs, score_noise)
    plt.plot(probs, score_filtered)
    plt.xlabel("Probability")
    plt.ylabel("Quality Score")
    plt.title("Quality Score vs Salt & Pepper Probability")
    plt.legend(["Original", "With Noise", "After Median"])
    plt.show()

def AnalysisRandomNoisingCurveVSGaussian(path: str, amplitude=10, sigma=0.6, steps=40, trials=3):
    if not os.path.exists(path):
        raise FileExistsError(f"{path} does not exist")

    amplitudes = np.linspace(0, amplitude, steps)

    var_noise = []
    var_filtered = []

    grad_noise = []
    grad_filtered = []

    score_noise = []
    score_filtered = []

    # Image originale
    f_original = DiscreteFunctionFromImage(path)

    var_original = LocalVariance(f_original)
    grad_original = GradientEnergy(f_original)
    score_original = QualityScore(f_original)

    g = GaussianDiscreteFunction(sigma)

    for a in amplitudes:

        var_n = grad_n = score_n = 0
        var_f = grad_f = score_f = 0

        # moyenne sur plusieurs tirages
        for _ in range(trials):

            f = DiscreteFunctionFromImage(path)

            # Bruit uniforme [-a, a]
            f.apply(randomNoising, int(-a), int(a))

            var_n += LocalVariance(f)
            grad_n += GradientEnergy(f)
            score_n += QualityScore(f)

            # Filtrage gaussien
            f_filtered = f.convolve(g)

            var_f += LocalVariance(f_filtered)
            grad_f += GradientEnergy(f_filtered)
            score_f += QualityScore(f_filtered)

        var_noise.append(var_n / trials)
        grad_noise.append(grad_n / trials)
        score_noise.append(score_n / trials)

        var_filtered.append(var_f / trials)
        grad_filtered.append(grad_f / trials)
        score_filtered.append(score_f / trials)

    # =========================
    # 1 - Variance
    # =========================
    plt.figure()
    plt.plot(amplitudes, [var_original]*len(amplitudes))
    plt.plot(amplitudes, var_noise)
    plt.plot(amplitudes, var_filtered)
    plt.xlabel("Noise Amplitude")
    plt.ylabel("Local Variance")
    plt.title("Local Variance vs Random Noise Amplitude")
    plt.legend(["Original", "With Noise", "After Gaussian"])
    plt.show()

    # =========================
    # 2 - Gradient
    # =========================
    plt.figure()
    plt.plot(amplitudes, [grad_original]*len(amplitudes))
    plt.plot(amplitudes, grad_noise)
    plt.plot(amplitudes, grad_filtered)
    plt.xlabel("Noise Amplitude")
    plt.ylabel("Gradient Energy")
    plt.title("Gradient Energy vs Random Noise Amplitude")
    plt.legend(["Original", "With Noise", "After Gaussian"])
    plt.show()

    # =========================
    # 3 - Quality Score
    # =========================
    plt.figure()
    plt.plot(amplitudes, [score_original]*len(amplitudes))
    plt.plot(amplitudes, score_noise)
    plt.plot(amplitudes, score_filtered)
    plt.xlabel("Noise Amplitude")
    plt.ylabel("Quality Score")
    plt.title("Quality Score vs Random Noise Amplitude")
    plt.legend(["Original", "With Noise", "After Gaussian"])
    plt.show()

# Filters

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

# Fourier

def InverseFourierTransformTest(path: str, rayonMax:int=-1):
    discreteImage=DiscreteFunctionFromImage(path)

    h, w = discreteImage.height, discreteImage.width

    start = time.time()
    FM_discreteImage = FourierTransform(discreteImage, rayonMax)
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
    numpyFM.resizeAmplitudeDiscreteFunction()

    Inverse_discreteImage.show()
    numpyFM.show()

    saveImageFromDiscreteFunction(Inverse_discreteImage, 'Pictures/Inverse_image.png')
    saveImageFromDiscreteFunction(numpyFM, "Pictures/Numpy_Inverse_image.png")

def FFT2DTest(path: str, completionMode:int):
    im=DiscreteFunctionFromImage(path)

    startTime=time.time()

    numpyF=ComplexDiscreteFunction(fft.fft2(im.kernel))
    print(time.time()-startTime)
    numpyFM=numpyF.getModule(True)
    numpyFM.resizeAmplitudeDiscreteFunction()
    numpyFMR=numpyFM.getRevolve()

    numpyFM.show()

    startTime=time.time()

    myF=ComplexDiscreteFunction(FFT2(im.kernel, completionMode))
    print(time.time()-startTime)
    myFM=myF.getModule(True)
    myFM.resizeAmplitudeDiscreteFunction()
    myFMR=myFM.getRevolve()

    myFM.show()

def IFFT2DTest(path:str, completionMode:int):
    im=DiscreteFunctionFromImage(path)

    myF=FFT2(im.kernel, completionMode)
    
    newIm=ComplexDiscreteFunction(IFFT2(myF, completionMode))
    newIm=newIm.getModule()
    newIm.resizeAmplitudeDiscreteFunction()

    newIm.show()

def FFTRadiusCutTest(path:str, noiseIntensity:float, radius:float):
    im=DiscreteFunctionFromImage(path)

    randomNoising(im, int(10**noiseIntensity), int(10**(noiseIntensity+1)))
    im.show()

    imF=ComplexDiscreteFunction(FFT2(im.kernel, 1))
    imFM=imF.getModule()
    imFM.resizeAmplitudeDiscreteFunction()
    imFM.show()

    imF.RadiusFilter(radius)
    imFM2=imF.getModule()
    imFM2.resizeAmplitudeDiscreteFunction()
    imFM2.show()

    im2=ComplexDiscreteFunction(IFFT2(imF.kernel, 1))
    im2=im2.getModule()
    im2.resizeAmplitudeDiscreteFunction()
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
