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
    image.apply_to_all(SaltAndPaperNoising, 0.1)
    image.show()
    f = GaussianDiscreteFunction(5, 1)
    d = 30.0
    #image.apply_to_all(DiscreteFunction.bilateralFilter, f, 50)
    image.apply_to_all(DiscreteFunction.medianFilter,1)

    image.show()

def TestMultiplesDeconvo(discreteFunction:DiscreteFunction):
    a=discreteFunction
    (a2, coordinates)=a.extend((2**ceil(log(a.width,2)), 2**ceil(log(a.height,2))))

    from random import random
    sigma=round(random()*2+1, 4)
    print("SIGMA:", sigma)

    b=GaussianDiscreteFunction(sigma)
    (b2,raf)=b.extend((a2.width, a2.height))
    b2=b2.getCentered()
    #b2.normalize()

    af=ComplexDiscreteFunction(FFT2(a2.kernel))
    b2f=ComplexDiscreteFunction(FFT2(b2.kernel))
    cf=af*b2f

    canva=Image.new("RGB", (a.width*10, a.height*10))

    x=[k/10 for k in range(10,30)]
    y={"yGradientEnergy":[],
       "yLocalVariance":[],
       "yEdgePreservation":[],
       "yHistogramSpread":[],
       "yRMS":[],
       "ySobelVariance":[],
       "yLaplacianVariance":[],
       "ySSIM":[],
       "yPSNR":[],
       "yMSE":[]}

    for k in range(100,300):
        d=GaussianDiscreteFunction(k/100)
        (d,raf)=d.extend((a2.width, a2.height))
        d2=d.getCentered()
        d2f=ComplexDiscreteFunction(FFT2(d2.kernel))

        ef=cf/d2f
        e=DiscreteFunction(IFFT2(ef.kernel))
        e.resize(coordinates=coordinates)

        eIm=getImageFromDiscreteFunction(e)
        canva.paste(eIm, ((((k-100)%10)*a.width),((k-9)//10)*a.height))
        #eIm.save("__pycache__/"+str(k/10)+".png")


        #y["yEdgePreservation"]=EdgePreservation(a, e)
        #y["yGradientEnergy"]=GradientEnergy(e)
        #y["yHistogramSpread"]=HistogramSpread(e)
        #y["yLaplacianVariance"]=LaplacianVariance(e)
        #y["yLocalVariance"]=LocalVariance(e)
        #y["yMSE"]=MSE(a, e)
        #y["yPSNR"]=PSNR(a, e)
        #y["yRMS"]=RMS(e)
        #y["ySobelVariance"]=SobelVariance(e)
        #y["ySSIM"]=SSIM(a, e)

        print("---------", (k-99)/(200)*100, "%")

    """for k in range(len(y.keys())):
        plt.subplot(4, 3, k+1)
        plt.plot(x, y[y.keys()[k]], "o-")
        plt.title(y.keys()[k])
    
    plt.show()"""

    canva.show()
    canva.save("__pycache__/canva.png")

#TestMultiplesDeconvo(DiscreteFunctionFromImage("Pictures/superman.png"))

def deconv_wiener(path):
    f = DiscreteFunctionFromImage(path).apply(RandomNoising, -30, 30)
    f.show()
    kernel = [[1 / 9, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 1 / 9, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1 / 9, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1 / 9, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 1 / 9, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1 / 9, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1 / 9, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 1 / 9, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 1 / 9]]
    kernel = DiscreteFunction(kernel)
    h = f.convolve(kernel)
    h.show()
    g = h.wienerDeconvolve(kernel,0.01)
    g.show()

    plt.show()

def gaussianDeconvolutionTest(originalDiscreteFunction:DiscreteFunction, extendedDiscreteFunction:DiscreteFunction, blurredDiscreteFunctionFourier:ComplexDiscreteFunction, coordinates:tuple, sigma:float):
    a=originalDiscreteFunction
    a2=extendedDiscreteFunction
    cf=blurredDiscreteFunctionFourier

    d=GaussianDiscreteFunction(sigma)
    (d,raf)=d.extend((a2.width, a2.height))
    d2=d.getCentered()
    d2f=ComplexDiscreteFunction(fft.fft2(d2.kernel).tolist())

    ef=cf/d2f
    e=ComplexDiscreteFunction(fft.ifft2(ef.kernel).tolist()).getModule(False)
    e.resize(coordinates=coordinates)

    y={"yGradientEnergy":0,
       "yLocalVariance":0,
       "yEdgePreservation":0,
       "yHistogramSpread":0,
       "yRMS":0,
       #"ySobelVariance":0,
       "yLaplacianVariance":0,
       "ySSIM":0,
       "yPSNR":0,
       "yMSE":0}

    y["yEdgePreservation"]=(EdgePreservation(a, e))
    y["yGradientEnergy"]=(GradientEnergy(e))
    y["yHistogramSpread"]=(HistogramSpread(e))
    y["yLaplacianVariance"]=(LaplacianVariance(e))
    y["yLocalVariance"]=(LocalVariance(e))
    y["yMSE"]=(MSE(a, e))
    y["yPSNR"]=(PSNR(a, e))
    y["yRMS"]=(RMS(e))
    #y["ySobelVariance"]=(SobelVariance(e))
    y["ySSIM"]=(SSIM(a, e))

    return getImageFromDiscreteFunction(e), y

def MultipleDeconvolutionBoostTest(discreteFunction:DiscreteFunction):
    from random import random
    startTime=time.time()

    sigma=round(random()*2.99+0.01, 3)
    sigma=1.5
    print("SIGMA:", sigma)

    a=discreteFunction
    a2=a
    coordinates = (0,0,a.width, a.height)#(a2, coordinates)=a.extend((2**ceil(log(a.width,2)), 2**ceil(log(a.height,2))))

    b=GaussianDiscreteFunction(sigma)
    (b2,raf)=b.extend((a2.width, a2.height))
    b2=b2.getCentered()
    #b2.normalize()

    af=ComplexDiscreteFunction(fft.fft2(a2.kernel).tolist()) 
    b2f=ComplexDiscreteFunction(fft.fft2(b2.kernel).tolist())
    cf=af*b2f

    c=ComplexDiscreteFunction(fft.ifft2(cf.kernel).tolist()).getModule(False)
    c.resize(coordinates=coordinates)
    #c.show()

    c=a

    canva=Image.new("RGB", (a.width*10,  a.height*10))

    x=[k/30 for k in range(1, 100)]
    y={"yGradientEnergy":[],
       "yLocalVariance":[],
       "yEdgePreservation":[],
       "yHistogramSpread":[],
       "yRMS":[],
       #"ySobelVariance":[],
       "yLaplacianVariance":[],
       "ySSIM":[],
       "yPSNR":[],
       "yMSE":[]}
    
    from multiprocessing import Pool
    mainPool=Pool()

    results=mainPool.starmap(gaussianDeconvolutionTest, [[a, a2, cf, coordinates, k/30] for k in range(1, 100)])

    for k in range(len(results)):
        canva.paste(results[k][0], (((k%10)*a.width),(k//10)*a.height))
        for key, value in results[k][1].items():
            y[key].append(value)

    #for k in range(len(y.keys())):
    #    plt.subplot(4, 3, k+1)
    #    plt.plot(x, y[list(y.keys())[k]], "o-")
    #    plt.title(list(y.keys())[k])
    
    #print("SIGMA:", sigma)
    #print("MAX SSIM:", y["ySSIM"].index(max(y["ySSIM"]))/30+0.1, "SSIM de :", max(y["ySSIM"]))

    canva.show()
    canva.save("__pycache__/canva.png")
    #print(time.time()-startTime)

    with open("__pycache__/deconvolution.csv", "w") as f:
        f.write("sigma;"+";".join(y.keys())+";;"+str(sigma)+"\n")
        for k in range(1,100):
            f.write(";".join([str(k)]+[str(y[key][k-1]).replace(".",",") for key in y.keys()])+"\n")
    
    #plt.show()
    DeconvolutionAnalyticsTest()
    
def DeconvolutionAnalyticsTest():
    
    x={"yGradientEnergy":[],
        "yLocalVariance":[],
        "yEdgePreservation":[],
        "yHistogramSpread":[],
        "yRMS":[],
        #"ySobelVariance":[],
        "yLaplacianVariance":[],
        "ySSIM":[],
        "yPSNR":[],
        "yMSE":[]}
    y={"yGradientEnergy":[],
        "yLocalVariance":[],
        "yEdgePreservation":[],
        "yHistogramSpread":[],
        "yRMS":[],
        #"ySobelVariance":[],
        "yLaplacianVariance":[],
        "ySSIM":[],
        "yPSNR":[],
        "yMSE":[]}

    with open("__pycache__/deconvolution.csv") as f:
        lines=f.readlines()
        header=lines[0].split(";")
        sigma=float(header[-1].replace(",", "."))

        for k in lines[1:]:
            k=k.split(";")
            for n in range(1, len(k)):
                y[header[n]].append(float(k[n].replace(",", ".")))

    statistics={}
    for key, value in y.items():
        stat=getStatistics(value)
        statistics[key]=stat
        #print(key, ":", stat)

    a=y["ySSIM"].copy()
    a.sort()
    SSIMMedian=a[len(a)//2]
    SSIMMedianCutoff=False
    
    alpha=1
    cutoffSSIM=0.1
    y2=y.copy()
    for key, value in y2.items():
        value2=[]
        for k in range(len(value)):
            if y["ySSIM"][k] >= cutoffSSIM: #statistics[key][0] - alpha*statistics[key][1] <= value[k] <= statistics[key][0] + alpha*statistics[key][1]:
                #if SSIMMedianCutoff and y["ySSIM"][k] < SSIMMedian: break
                value2.append(value[k])
                x[key].append((k+1)/30)
        y2[key]=value2

    for k in range(len(y.keys())):
        plt.subplot(4, 3, k+1)
        plt.plot(x[list(x.keys())[k]], y2[list(y2.keys())[k]], "o-")
        try: plt.plot([sigma-0.0001, sigma, sigma+0.0001], [min(y2[list(y2.keys())[k]]), max(y2[list(y2.keys())[k]]), min(y2[list(y2.keys())[k]])], "r-")
        except: pass
        plt.title(list(y.keys())[k])

    plt.show()

def PeriodicNoiseRemovalTest(discreteFunction:DiscreteFunction):
    from math import cos
    
    a=discreteFunction
    
    b=a.copy()
    """for k in range(b.width):
        for n in range(b.height):
            b[k,n]-=cos(n)*70+58"""
    #b.show()
    
    bf=ComplexDiscreteFunction(fft.fft2(b.kernel))
    
    canva=Image.new("RGB", (a.width*10, a.height*5))
    
    x=[k/100 for k in range(1,50)]
    y={"yGradientEnergy":[],
       "yLocalVariance":[],
       "yEdgePreservation":[],
       "yHistogramSpread":[],
       "yRMS":[],
       "yLaplacianVariance":[],
       "ySSIM":[],
       "yPSNR":[],
       "yMSE":[]}
    
    for k in range(1,50):
        bf2=bf.copy()
        bf2.AmplitudeCutFilter(k/100, 4, False)
    
        b2=ComplexDiscreteFunction(fft.ifft2(bf2.kernel).tolist()).getModule(False)
        b2Im=getImageFromDiscreteFunction(b2)
    
        canva.paste(b2Im, ((((k-1)%10)*a.width),((k-1)//10)*a.height))
    
        #eIm.save("__pycache__/"+str(k/10)+".png")
        y["yEdgePreservation"].append(EdgePreservation(b, b2))
        y["yGradientEnergy"].append(GradientEnergy(b2))
        y["yHistogramSpread"].append(HistogramSpread(b2))
        y["yLaplacianVariance"].append(LaplacianVariance(b2))
        y["yLocalVariance"].append(LocalVariance(b2))
        y["yMSE"].append(MSE(b, b2))
        y["yPSNR"].append(PSNR(b, b2))
        y["yRMS"].append(RMS(b2))
        y["ySSIM"].append(SSIM(b, b2))
    
        print("---------", ((k+1)*100)//(50), "%")
    
    for k in range(len(y.keys())):
        plt.subplot(4, 3, k+1)
        plt.plot(x, y[list(y.keys())[k]], "o-")
        plt.title(list(y.keys())[k])
    
    with open("__pycache__/periodicNoise.csv", "w") as f:
        f.write("cutoff;"+";".join(y.keys())+"\n")
        for k in range(1,50):
            f.write(";".join([str(k)]+[str(y[key][k-1]).replace(".",",") for key in y.keys()])+"\n")
    
    canva.show()
    canva.save("__pycache__/canva.png")
    
    #plt.show()
    PeriodicNoiseAnalyticsTest()

def PeriodicNoiseAnalyticsTest():
    
    x={"yGradientEnergy":[],
        "yLocalVariance":[],
        "yEdgePreservation":[],
        "yHistogramSpread":[],
        "yRMS":[],
        #"ySobelVariance":[],
        "yLaplacianVariance":[],
        "ySSIM":[],
        "yPSNR":[],
        "yMSE":[]}
    y={"yGradientEnergy":[],
        "yLocalVariance":[],
        "yEdgePreservation":[],
        "yHistogramSpread":[],
        "yRMS":[],
        #"ySobelVariance":[],
        "yLaplacianVariance":[],
        "ySSIM":[],
        "yPSNR":[],
        "yMSE":[]}

    with open("__pycache__/periodicNoise.csv") as f:
        lines=f.readlines()
        header=lines[0].split(";")

        for k in lines[1:]:
            k=k.split(";")
            for n in range(1, len(k)):
                y[header[n].rstrip()].append(float(k[n].replace(",", ".")))

    statistics={}
    for key, value in y.items():
        stat=getStatistics(value)
        statistics[key]=stat
        print(key, ":", stat)

    alpha=1
    cutoffSSIM=0.8
    y2=y.copy()
    for key, value in y2.items():
        value2=[]
        for k in range(len(value)):
            if y["ySSIM"][k] >= cutoffSSIM: #statistics[key][0] - alpha*statistics[key][1] <= value[k] <= statistics[key][0] + alpha*statistics[key][1]:
                value2.append(value[k])
                x[key].append(k/100+0.01)
        y2[key]=value2

    for k in range(len(y.keys())):
        plt.subplot(4, 3, k+1)
        plt.plot(x[list(x.keys())[k]], y2[list(y2.keys())[k]], "o-")
        plt.title(list(y.keys())[k])

    plt.show()

def PeriodicNoiseRemoval(discreteFunction:DiscreteFunction) -> DiscreteFunction:
    a=discreteFunction
    af=ComplexDiscreteFunction(fft.fft2(a.kernel).tolist())

    a2s=[]
    SSIMs=[]

    for k in range(1,50):
        af2=af.copy()
        af2.AmplitudeCutFilter(k/100, 4, False)
    
        a2=ComplexDiscreteFunction(fft.ifft2(af2.kernel).tolist()).getModule(False)
        a2s.append(a2)

        SSIMs.append(SSIM(a, a2))

        print(k*2, "%")

        if len(SSIMs) >= 2 and SSIMs[-1] == SSIMs[-2]:
            return a2
    
    minSSIMGradIndex=0
    minSSIMGrad=float("+inf")

    for k in range(len(SSIMs)-1):
        SSIMGrad=SSIMs[k+1]-SSIMs[k]
        if SSIMGrad < minSSIMGrad:
            minSSIMGrad=SSIMGrad
            minSSIMGradIndex=k+1
    
    return a2s[minSSIMGradIndex]

def GaussianDeconvolution(discreteFunction, discreteFunctionFourier, sigma):
    a=discreteFunction
    af=discreteFunctionFourier

    d=GaussianDiscreteFunction(sigma)
    (d,raf)=d.extend((af.width, af.height))
    d2=d.getCentered()
    d2f=ComplexDiscreteFunction(fft.fft2(d2.kernel).tolist())

    ef=af/d2f
    e=ComplexDiscreteFunction(fft.ifft2(ef.kernel).tolist()).getModule(False)

    return e, SSIM(a, e), HistogramSpread(e), sigma

def GaussianNoiseRemoval(discreteFunction:DiscreteFunction, minValue:float, maxValue:float, steps:int):
    assert minValue < maxValue

    a=discreteFunction
    af=ComplexDiscreteFunction(fft.fft2(a.kernel).tolist())

    increments=round((maxValue-minValue)/(steps-1), 4)
    sigmas=[round(minValue+k*increments, 4) for k in range(steps)]

    pool=Pool()
    results=pool.starmap(GaussianDeconvolution, [(a, af, sigmas[k]) for k in range(len(sigmas))])
    


    SSIMs=[]
    for k in range(len(results)):
        if results[k][1] >= 0:
            SSIMs.append(results[k][1])
    SSIMs.sort()
    medianSSIM=SSIMs[len(SSIMs)//2]


    goodSSIMs=[]
    for k in range(len(results)):
        if results[k][1] >= medianSSIM:
            goodSSIMs.append(results[k])
    
    goodSSIMs.sort(key=lambda x: x[2])

    sideSize=ceil(sqrt(steps))
    canva=Image.new("RGB", (a.width*sideSize,  a.height*sideSize))
    for k in range(len(results)):
        canva.paste(getImageFromDiscreteFunction(results[k][0]), (((k%sideSize)*a.width),(k//sideSize)*a.height))
    
    canva.show()
    canva.save("__pycache__/canva.png")

    print(goodSSIMs[-1][3])
    return goodSSIMs[-1][0]

def WienerDeconvolution(discreteFunction:DiscreteFunction, kernel:DiscreteFunction, minK:float, maxK:float, Ksteps:int):
    a=DiscreteFunctionFromImage("pictures/toto.png")

    c=discreteFunction
    cf=ComplexDiscreteFunction(fft.fft2(c.kernel).tolist())

    b=kernel
    (b, raf)=b.extend((c.width, c.height))
    b=b.getCentered()
    b.normalize()
    bf=ComplexDiscreteFunction(fft.fft2(b.kernel).tolist())

    increments=round((maxK-minK)/(Ksteps-1), 4)
    values=[round(minK+k*increments, 4) for k in range(Ksteps)]
    print(values)

    sideSize=ceil(sqrt(Ksteps))
    canva=Image.new("RGB", (c.width*sideSize,  c.height*sideSize))
    for k in range(len(values)):
        ef=cf.wienerDeconvolve(bf, values[k])
        e=ComplexDiscreteFunction(fft.ifft2(ef.kernel).tolist()).getModule(False)

        canva.paste(getImageFromDiscreteFunction(e), (((k%sideSize)*c.width),(k//sideSize)*c.height))

    canva.show()
    c.show()


"""
if __name__ == "__main__": 
    from random import random

    sigma=round(random()*3+0.1, 2)

    a=DiscreteFunctionFromImage("Pictures/elephant.png")
    af=ComplexDiscreteFunction(fft.fft2(a.kernel).tolist())

    b=GaussianDiscreteFunction(sigma)
    (b, raf)=b.extend((a.width, a.height))
    b2=b.getCentered()
    b2f=ComplexDiscreteFunction(fft.fft2(b2.kernel).tolist())

    cf=af*b2f
    c=ComplexDiscreteFunction(fft.ifft2(cf.kernel).tolist()).getModule(False)

    c.show()

    #MultipleDeconvolutionBoostTest(DiscreteFunctionFromImage("Pictures/elephant.png"))
    result=GaussianNoiseRemoval(c, 0.1, 3, 300)
    result.show()

    print(sigma)"""
#TestMultiplesDeconvo(DiscreteFunctionFromImage("Pictures/superman.png"))
#DeconvolutionAnalyticsTest()

#PeriodicNoiseAnalyticsTest()
#PeriodicNoiseRemovalTest(DiscreteFunctionFromImage("Pictures/rubiks.png"))

#GaussianNoiseRemoval(None, 0.1,1,10)

"""
if __name__=="__main__":
    #MultipleDeconvolutionBoostTest(DiscreteFunctionFromImage("pictures/blue lobster.jpg"))
    a=DiscreteFunctionFromImage("pictures/blue lobster.jpg")
    af=ComplexDiscreteFunction(fft.fft2(a.kernel).tolist())

    b=GaussianDiscreteFunction(1.5)
    (b, coordinates)=b.extend((a.width, a.height))
    bf=ComplexDiscreteFunction(fft.fft2(b.kernel).tolist())

    cf=af*bf
    c=ComplexDiscreteFunction(fft.ifft2(cf.kernel).tolist()).getModule(False)

    GaussianNoiseRemoval(c, 0.1, 3.0, 100)"""

"""
a=DiscreteFunctionFromImage("pictures/blue lobster.jpg")
af=ComplexDiscreteFunction(fft.fft2(a.kernel).tolist())

b=GaussianDiscreteFunction(1)
(b, coordinates)=b.extend((a.width, a.height))
bf=ComplexDiscreteFunction(fft.fft2(b.kernel).tolist())

cf=af*bf
c=ComplexDiscreteFunction(fft.ifft2(cf.kernel).tolist()).getModule(False)

sigmas=[0.5, 1.5]
for k in range(len(sigmas)):
    d=GaussianDiscreteFunction(sigmas[k])
    (d, raf)=d.extend((c.width, c.height))
    d=d.getCentered()
    
    df=ComplexDiscreteFunction(fft.fft2(d.kernel).tolist())

    ef=cf/df
    e=ComplexDiscreteFunction(fft.ifft2(ef.kernel).tolist()).getModule(False)
    getImageFromDiscreteFunction(e.getCentered()).save(f"__pycache__/blueLobsterDeconvolved{k+4}.png")"""


def deconvo(af:ComplexDiscreteFunction, value:float) -> DiscreteFunction:
    #b=GaussianDiscreteFunction(value)
    b=DiscreteFunction([[1 for k in range(2*int(value)+1)] for n in range(2*int(value)+1)])
    (b,raf)=b.extend((af.width, af.height))
    b2=b.getCentered()
    b2.normalize()

    b2f=ComplexDiscreteFunction(fft.fft2(b2.kernel).tolist())

    cf=af.wienerDeconvolve(b2f, 0.01)
    c=ComplexDiscreteFunction(fft.ifft2(cf.kernel).tolist()).getModule(False)
    
    return c


if __name__ == "__main__":
    a=DiscreteFunctionFromImage("pictures/photo_2_redim.png")
    a=a.medianFilter(1)
    #a=DiscreteFunctionFromImage("pictures/toto.png").convolve(DiscreteFunction([[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]]))
    af=ComplexDiscreteFunction(fft.fft2(a.kernel).tolist())

    minValue=1
    maxValue=20
    steps=20

    increments=round((maxValue-minValue)/(steps-1), 4)
    values=[round(minValue+k*increments, 4) for k in range(steps)]
    print(values)

    pool=Pool()

    results=pool.starmap(deconvo, [(af, k) for k in values])

    sideSize=ceil(sqrt(steps))
    canva=Image.new("RGB", (a.width*sideSize,  a.height*sideSize))
    for k in range(len(results)):
        getImageFromDiscreteFunction(results[k]).save(f"__pycache__/multipleResults/{values[k]}.png")
        canva.paste(getImageFromDiscreteFunction(results[k]), (((k%sideSize)*a.width),(k//sideSize)*a.height))
    
    canva.save("__pycache__/canva.png")
    canva.show()