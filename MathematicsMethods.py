from math import atan2, pi, e, log
from multiprocessing import Pool

def arg(z:complex):
    return atan2(z.real, z.imag)

def GaussianKernel(sigma: float, size: int) -> list[list[float]]:
        from math import pi, exp

        center = size // 2
        kernel = [[0.0] * size for _ in range(size)]
        norm = 0.0

        for j in range(size):
            for i in range(size):
                dx = i - center
                dy = j - center
                value = (1 / (2 * pi * sigma**2)) * exp(
                    -(dx**2 + dy**2) / (2 * sigma**2)
                )
                kernel[j][i] = value
                norm += value

        # normalize
        for j in range(size):
            for i in range(size):
                kernel[j][i] /= norm

        return kernel

def FourierTransform(kernel:list[list[float]]) -> list[list[complex]]: # ATTENTION ne marche pas avec des non puissances de 2
    mat:list[list] = []

    for q in range(len(kernel)):
        mat.append([])
        print(round(q/len(kernel)*100, 1), "%")
        for p in range(len(kernel[0])):
            value=0
            for m in range(len(kernel[0])):
                for n in range(len(kernel)):
                    theta=-2j*pi*(p*m/len(kernel[0])+q*n/len(kernel))
                    value+=kernel[n][m]*e**theta
            
            mat[-1].append(value)
    
    return mat

def DFT(kernel: list[list[float]]): # ATTENTION ne marche pas avec des non puissances de 2
    mat=[]

    for q in range(len(kernel)):
        mat.append([])
        for p in range(len(kernel[0])):
            value=0
            for m in range(len(kernel[0])):
                value+=kernel[q][m]*e**(-2j*pi*p*m/len(kernel[0]))
            mat[-1].append(value)
    
    mat2=[[] for k in range(len(mat))]
    for p in range(len(mat[0])):
        for q in range(len(mat)):
            value=0
            for n in range(len(mat)):
                value+=mat[n][p]*e**(-2j*pi*q*n/len(kernel))
            mat2[q].append(value)
    
    return mat2


def InverseFourierTransform(kernel:list[list[complex]], rayonMax: int = -1) -> list[list[float]]:
    mat: list[list] = []
    N = len(kernel[0]) * len(kernel)

    for q in range(len(kernel)):
        mat.append([])
        print(round(q / len(kernel) * 100, 1), "%")
        for p in range(len(kernel[0])):
            value = 0
            if rayonMax >= 0:
                for m in range(p - rayonMax, p + rayonMax + 1):
                    for n in range(q - rayonMax, q + rayonMax + 1):
                        theta = 2 * pi * (p * (rayonMax + m) / (2 * rayonMax + 1) + q * (rayonMax + n) / (2 * rayonMax + 1))
                        value += kernel[n][m] * e ** (theta * 1j)
            else:
                for m in range(len(kernel[0])):
                    for n in range(len(kernel)):
                        theta = 2 * pi * (p * m / len(kernel[0]) + q * n / len(kernel))
                        value += kernel[n][m] * e ** (theta * 1j)

            mat[q].append(value/N)

    return mat

def completeTo2(liste:list, mode:int) -> list:
    if log(len(liste), 2)!=int(log(len(liste), 2)):
        match mode:
            case 1: 
                if liste is list: liste+=[[0 for n in range(len(liste[0]))] for k in range(2**(int(log(len(liste), 2))+1)-len(liste))]
                else: liste+=[0]*(2**(int(log(len(liste), 2))+1)-len(liste))
            case 2: liste+=[liste[k%len(liste)] for k in range(2**(int(log(len(liste), 2))+1)-len(liste))]
    return liste

def FFT(data:list, completionMode:int=0) -> list[complex]:
    if len(data)==1: return data
    if completionMode!=0: data=completeTo2(data, completionMode)
    
    theta=e**(-2*pi*1j/len(data))
    fft_gauche=FFT([data[2*k] for k in range(len(data)//2)])
    fft_droite=FFT([data[2*k+1] for k in range(len(data)//2)])
    for k in range(len(fft_droite)):
        fft_droite[k]*=theta**k

    mat=[]
    for k in range(len(data)//2):
        mat.append(fft_gauche[k]+fft_droite[k])
    for k in range(len(data)//2):
        mat.append(fft_gauche[k]-fft_droite[k])
    
    return mat

def FFT2(kernel:list[list[complex]], completionMode:int=2) -> list[list[complex]]:
    horizontalFFTKernel=[]
    
    for k in range(len(kernel)):
        horizontalFFTKernel.append(FFT(kernel[k], completionMode))
    
    horizontalFFTKernel=completeTo2(horizontalFFTKernel, completionMode)
    finishedFFTKernel=[[] for k in range(len(horizontalFFTKernel))]

    for k in range(len(horizontalFFTKernel[0])):
        currentListFFT=FFT([horizontalFFTKernel[n][k] for n in range(len(horizontalFFTKernel))], completionMode)
        for n in range(len(currentListFFT)):
            finishedFFTKernel[n].append(currentListFFT[n])
    
    return finishedFFTKernel

def FFT2Boost(kernel:list[list[complex]], completionMode:int=2, pool=None):
    if not pool: pool=Pool()
    horizontalFFTKernel=[]

    horizontalFFTKernel = pool.starmap(FFT, [(kernel[k], completionMode) for k in range(len(kernel))])

    horizontalFFTKernel=completeTo2(horizontalFFTKernel, completionMode)
    finishedFFTKernel = [[] for k in range(len(horizontalFFTKernel))]

    results=pool.starmap(FFT, [([horizontalFFTKernel[n][k] for n in range(len(horizontalFFTKernel))], completionMode) for k in range(len(horizontalFFTKernel[0]))])

    for k in range(len(horizontalFFTKernel[0])):
        for n in range(len(results[k])):
            finishedFFTKernel[n].append(results[k][n])
    
    return finishedFFTKernel

def sectionnedFFT2Base(kernel):
    return [FFT(k, 2) for k in kernel]

def sectionnedFFT2Boost(kernel:list[list[float]], numberOfSections:int):
    from multiprocessing import Pool

    pool=Pool()
    horizontalFFTKernel=[]

    newKernel=[]
    for k in range(0, len(kernel), len(kernel)//numberOfSections):
        newKernel.append(kernel[k:k+len(kernel)//numberOfSections])
    
    for section in pool.map(sectionnedFFT2Base, newKernel):
        horizontalFFTKernel+=section
    
    transposed=[[] for k in range(len(horizontalFFTKernel[0]))]

    for k in range(len(horizontalFFTKernel)):
        for n in range(len(horizontalFFTKernel[k])):
            transposed[n].append(horizontalFFTKernel[k][n])
    
    newKernel=[]
    for k in range(0, len(transposed), len(transposed)//numberOfSections):
        newKernel.append(transposed[k:k+len(transposed)//numberOfSections])

    results=[]
    for section in pool.map(sectionnedFFT2Base, newKernel):
        results+=section
    
    finishedFFTKernel=[[] for k in range(len(results[0]))]
    for k in range(len(results)):
        for n in range(len(results[k])):
            finishedFFTKernel[n].append(results[k][n])
    
    return finishedFFTKernel


def IFFT(data:list, completionMode:int, floatResult:bool=False, firstTime:bool=False) -> list[float]:
    if len(data)==1: 
        if firstTime and floatResult: data[0]=abs(data[0])
        return data
    if firstTime: data=completeTo2(data, completionMode)

    mat=[]
    for k in range(len(data)//2):
        mat.append((data[k]+data[len(data)//2+k])/2)
    for k in range(len(data)//2):
        mat.append((data[k]-data[len(data)//2+k])*e**(2*pi*1j*k/len(data))/2)

    ifft_gauche=IFFT(mat[:len(data)//2], completionMode)
    ifft_droite=IFFT(mat[len(data)//2:], completionMode)

    mat=[]
    for k in range(len(data)//2):
        mat.append(ifft_gauche[k])
        mat.append(ifft_droite[k])

    if firstTime and floatResult: mat=[abs(k) for k in mat]

    return mat

def IFFT2(kernel:list[list[complex]], completionMode:int=2) -> list[list[float]]:
    horizontalFFTKernel=[]
    finishedFFTKernel=[]

    for k in range(len(kernel)):
        horizontalFFTKernel.append(IFFT(kernel[k], completionMode, False, True))
        finishedFFTKernel.append([])

    horizontalFFTKernel=completeTo2(horizontalFFTKernel, completionMode)
    finishedFFTKernel=completeTo2(finishedFFTKernel, 2)

    for k in range(len(horizontalFFTKernel[0])):
        currentListIFFT=IFFT([horizontalFFTKernel[n][k] for n in range(len(horizontalFFTKernel))], completionMode, True, True)
        for n in range(len(currentListIFFT)):
            finishedFFTKernel[n].append(currentListIFFT[n])

    return finishedFFTKernel

def getEcartRel(ref:list, test:list) -> float:
    ecartMoy=0
    for k in range(len(ref)):
        try:
            if ref[k]!=0: ecartMoy+=abs((ref[k]-test[k])/ref[k])
            else: ecartMoy+=abs(test[k])
        except: 
            ecartMoy+=getEcartRel(ref[k], test[k])
    return float(ecartMoy)/len(ref)