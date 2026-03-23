from math import log, sqrt
from ImageMethods import *
from MathematicsMethods import *

class DiscreteFunction:
    def __init__(self, kernel:list[list[float]], x:int = 0, y:int = 0):
        self.kernel: list[list[float]] = kernel
        self.width: int = len(kernel[0])
        self.height: int = len(kernel)
        self.x: int = x
        self.y: int = y

    def __getitem__(self, item:tuple):
        #item[0] = x
        #item[1] = y
        if not (self.x <= item[0] < self.x + self.width):
            return 0
        if not (self.y <= item[1] < self.y + self.height):
            return 0
        return self.kernel[item[1] - self.y][item[0] - self.x]
    
    def __setitem__(self, item:tuple, value: float):
        #item[0] = x
        #item[1] = y
        if not (self.x <= item[0] < self.x + self.width):
            raise IndexError
        if not (self.y <= item[1] < self.y + self.height):
            raise IndexError
        self.kernel[item[1] - self.y][item[0] - self.x] = value

    def __add__(self, value):
        if type(value) == int:
            return DiscreteFunction([[self[i, j] + value for i in range(self.width)] for j in range(self.height)], x=self.x, y=self.y)
        elif type(value) == DiscreteFunction:
            xMax = max(self.width + self.x , value.width + value.x)
            yMax = max(self.height + self.y, value.height + value.y)
            xMin = min (self.x, value.x)
            yMin = min (self.y, value.y)
            return DiscreteFunction([[self[i, j] + value[i, j] for i in range(xMin, xMax)] for j in range(yMin, yMax)], x=xMin, y=yMin)
        else:
            raise TypeError(f"unsupported operand type(s) for '+' : 'DiscreteFunction' and '{ type(value).__name__}'")

    def __mul__(self, value):
        if type(value) == float or type(value) == int:
            return DiscreteFunction([[self[i, j] * value for i in range(self.width)] for j in range(self.height)], x=self.x, y=self.y)
        elif type(value) == DiscreteFunction:
            xMax = min(self.width + self.x , value.width + value.x)
            yMax = min(self.height + self.y, value.height + value.y)
            xMin = max (self.x, value.x)
            yMin = max (self.y, value.y)
            return DiscreteFunction([[self[i, j] * value[i, j] for i in range(xMin, xMax)] for j in range(yMin, yMax)], x=xMin, y=yMin)
        else:
            raise TypeError(f"unsupported operand type(s) for '*' : 'DiscreteFunction' and '{ type(value).__name__}'")
        
    def __sub__(self, value):
        if type(value) == int:
            return DiscreteFunction([[self[i, j] - value for i in range(self.width)] for j in range(self.height)], x=self.x, y=self.y)
        elif type(value) == DiscreteFunction:
            xMax = max(self.width + self.x , value.width + value.x)
            yMax = max(self.height + self.y, value.height + value.y)
            xMin = min (self.x, value.x)
            yMin = min (self.y, value.y)
            return DiscreteFunction([[self[i, j] - value[i, j] for i in range(xMin, xMax)] for j in range(yMin, yMax)], x=xMin, y=yMin)
        else:
            raise TypeError(f"unsupported operand type(s) for '-' : 'DiscreteFunction' and '{ type(value).__name__}'")
        
    def __eq__(self, value):
        if type(value) != DiscreteFunction:
            return False

        if self.height != value.height or self.width != value.width:
            return False
        
        if self.x != value.x or self.y != value.y:
            return False

        for i in range(self.width):
            for j in range(self.height):
                if self[i, j] != value[i, j]:
                    return False
                
        return True
    
    def convolve(self, other: type["DiscreteFunction"]):
        if not isinstance(other, DiscreteFunction):
            raise TypeError

        g = DiscreteFunction(
            [[0]*self.width for _ in range(self.height)],
            x=self.x,
            y=self.y
        )
        for i in range(self.width):
            #print((i+1)*100//self.width, "%")
            for j in range(self.height):
                g[i, j] = sum(
                    self[i - m, j - n] * other[m, n]
                    for m in range(other.width)
                    for n in range(other.height)
                )
        return g

    def adaptativeGaussianConvolution(self, other: "GaussianDiscreteFunction", diff: float):
        if not isinstance(other, GaussianDiscreteFunction):
            raise TypeError

        g = DiscreteFunction(
            [[0]*self.width for _ in range(self.height)],
            x=self.x,
            y=self.y
        )

        centerX = other.width // 2
        centerY = other.height // 2

        for i in range(self.width):
            for j in range(self.height):

                weighted_sum = 0
                norm = 0

                for m in range(other.width):
                    for n in range(other.height):

                        dx = m - centerX
                        dy = n - centerY

                        neighbor = self[i + dx, j + dy]
                        center_pixel = self[i, j]

                        if abs(center_pixel - neighbor) < diff:
                            weight = other[m, n]
                            weighted_sum += neighbor * weight
                            norm += weight

                if norm != 0:
                    g[i, j] = weighted_sum / norm
                else:
                    g[i, j] = self[i, j]

        return g

    def bilateralFilter(self, spatialKernel: "GaussianDiscreteFunction", sigma_r: float):
        from math import exp

        g = DiscreteFunction(
            [[0]*self.width for _ in range(self.height)],
            x=self.x,
            y=self.y
        )

        centerX = spatialKernel.width // 2
        centerY = spatialKernel.height // 2

        for i in range(self.width):
            for j in range(self.height):

                weighted_sum = 0
                norm = 0
                center = self[i, j]

                for m in range(spatialKernel.width):
                    for n in range(spatialKernel.height):

                        dx = m - centerX
                        dy = n - centerY

                        neighbor = self[i + dx, j + dy]

                        intensity_diff = neighbor - center
                        w_r = exp(-(intensity_diff**2) / (2 * sigma_r**2))

                        weight = spatialKernel[m, n] * w_r

                        weighted_sum += neighbor * weight
                        norm += weight

                if norm != 0:
                    g[i, j] = weighted_sum / norm
                else:
                    g[i, j] = center

        return g

    def normalize(self):
        norm = sum(sum(row) for row in self.kernel)

        for i in range(self.width):
            for j in range(self.height):
                norm+=abs(self[i,j])
        
        if norm == 0:
            raise ValueError("Cannot normalize a kernel with zero sum")

        self.kernel = [[value / norm for value in row] for row in self.kernel]

    def resizeAmplitude(self, minValue:int=0, maxValue:int=255):
        minV=float("inf")
        maxV=float("-inf")

        for i in range(self.width):
            for j in range(self.height):
                if self[i,j]>maxV: maxV=self[i,j]
                if self[i,j]<minV and self[i,j]!=float("-inf"): minV=self[i,j]

        if maxV!=minV:
            for i in range(self.width):
                for j in range(self.height):
                    self[i,j]=(self[i,j]-minV)*(maxValue-minValue)/(maxV-minV)+minValue
    
    def medianFilter(self, radius: int = 1):
        newKernel = [[0]*self.width for _ in range(self.height)]
        radius=int(radius)

        for i in range(self.width):
            for j in range(self.height):
                neighbours = [
                    self[i+k, j+n]
                    for k in range(-radius, radius+1)
                    for n in range(-radius, radius+1)
                ]
                neighbours.sort()
                newKernel[j][i] = neighbours[(len(neighbours)-1)//2]

        self.kernel = newKernel
        return self

    def Expectation(self):
        N = self.width * self.height
        total = 0
        for j in range(self.height):
            for i in range(self.width):
                total += self[i, j]
        return total / N

    def Variance (self):
        mean = self.Expectation()
        N = self.width * self.height

        variance = sum(
            (self[i, j] - mean)**2
            for j in range(self.height)
            for i in range(self.width)
        ) / N

        return variance

    def StandardDeviation(self):
        from math import sqrt
        return sqrt(self.Variance())

    def apply(self, func, *args, **kwargs):
        func(self, *args, **kwargs)
        return self

    def copy(self):
        import copy
        return DiscreteFunction(copy.deepcopy(self.kernel), self.x, self.y)

    def show(self):
        image = getImageFromDiscreteFunction(self)
        image.show()

    def getHistorigram(self):
        histogram=[0 for k in range(256)]

        for i in range(self.width):
            for j in range(self.height):
                value = int(round(self[i,j]))
                if value <= 0: 
                    histogram[0]+=1
                elif value >= 255: 
                    histogram[255]+=1
                else: 
                    histogram[value]+=1

        return histogram

    def getCentered(self):
        kernel=self.kernel.copy()

        if self.width%2==1:
            for j in range(self.height):
                kernel[j].append(0)
        if self.height%2==1:
            kernel.append([0 for k in range(self.width)])
        
        newDF=DiscreteFunction(kernel)
        
        mat=[[0 for i in range(newDF.width)] for j in range(newDF.height)]
        for i in range(newDF.width):
            for j in range(newDF.height):
                mat[j][i]=newDF[fftShiftIndex(newDF.width, newDF.height, (i,j))]

        return DiscreteFunction(mat)

    def extend(self, newSize):
        kernel=self.kernel.copy()

        if self.width%2==0:
            if newSize[0]%2==0:
                a=(newSize[0]-len(kernel[0]))//2
                x=a

                for j in range(len(kernel)):
                    for k in range(a):
                        kernel[j].append(0)
                        kernel[j].insert(0, 0)
            else:
                a=(newSize[0]-len(kernel[0])-1)//2
                x=a
                b=newSize[0]-2

                for j in range(len(kernel)):
                    for k in range(a):
                        kernel[j].append(0)
                        kernel[j].insert(0, 0)
                    for k in range(b, 2, -1):
                        kernel[j][k]=(kernel[j][k]+kernel[j][k-1])/2
                    kernel[j].append(0)
        else:
            if newSize[0]%2==1:
                a=(newSize[0]-len(kernel[0])+1)//2
                x=a

                for j in range(len(kernel)):
                    for k in range(a):
                        kernel[j].append(0)
                        kernel[j].insert(0, 0)
            else:
                a=(newSize[0]-len(kernel[0])-1)//2
                x=a
                b=newSize[0]-2

                for j in range(len(kernel)):
                    for k in range(a):
                        kernel[j].append(0)
                        kernel[j].insert(0, 0)
                    for k in range(b, 2, -1):
                        kernel[j][k]=(kernel[j][k]+kernel[j][k-1])/2
                    kernel[j].append(0)

        if self.height%2==0:
            if newSize[1]%2==0:
                a=(newSize[1]-len(kernel))//2
                y=a

                for k in range(a):
                    kernel.append([0 for j in range(len(kernel[0]))])
                    kernel.insert(0, [0 for j in range(len(kernel[0]))])
            else:
                a=(newSize[1]-len(kernel)-1)//2
                y=a
                b=newSize[1]-2

                for k in range(a):
                    kernel.append([0 for j in range(len(kernel[0]))])
                    kernel.insert(0, [0 for j in range(len(kernel[0]))])
                for k in range(b, 2, -1):
                    for j in range(len(kernel[0])):
                        kernel[k][j]=(kernel[k][j]+kernel[k-1][j])/2
                kernel.append([0 for j in range(len(kernel[0]))])
        else:
            if newSize[1]%2==1:
                a=(newSize[1]-len(kernel)+1)//2
                y=a

                for k in range(a):
                    kernel.append([0 for j in range(len(kernel[0]))])
                    kernel.insert(0, [0 for j in range(len(kernel[0]))])
            else:
                a=(newSize[1]-len(kernel)-1)//2
                y=a
                b=newSize[1]-2

                for k in range(a):
                    kernel.append([0 for j in range(len(kernel[0]))])
                    kernel.insert(0, [0 for j in range(len(kernel[0]))])
                for k in range(b, 2, -1):
                    for j in range(len(kernel[0])):
                        kernel[k][j]=(kernel[k][j]+kernel[k-1][j])/2
                kernel.append([0 for j in range(len(kernel[0]))])

        coordinates=(x,y,self.width,self.height)
        return DiscreteFunction(kernel), coordinates

    def resize(self, x=0, y=0, width=0, height=0, coordinates=None):
        if coordinates:
            x,y,width,height=coordinates[0], coordinates[1], coordinates[2], coordinates[3]

        for k in range(self.height):
            for j in range(x):
                self.kernel[k].pop(0)
        for k in range(y):
            self.kernel.pop(0)
        
        for k in range(self.height):
            for j in range(len(self.kernel[0])-width):
                self.kernel[j].pop()
        for k in range(len(self.kernel)-height):
            self.kernel.pop()
        
        self.width=width
        self.height=height


"""
    def wienerDeconvolve(self, kernel: "DiscreteFunction", K: float = 0.01):

        #Déconvolution de Wiener pour retirer un flou.

        #kernel : noyau de convolution (PSF)
        #K : paramètre de régularisation bruit/signal


        from MathematicsMethods import FFT2Boost

        # FFT de l'image floue
        F = DiscreteFunction(FFT2Boost(self.kernel))

        # FFT du noyau (mis à la taille de l'image)
        #H = kernel.pad(self.width, self.height).fft2()

        # Filtre de Wiener
        for i in range(F.width):
            for j in range(F.height):

                h = H[i, j]

                if abs(h) == 0:
                    F[i, j] = 0
                else:
                    F[i, j] = F[i, j] * h.conjugate() / (abs(h)**2 + K)

        # Retour domaine spatial
        deconv = F.ifft2()

        # Garder la partie réelle
        deconv = deconv.real()

        # Normalisation
        deconv.resizeAmplitude(0, 255)

        return deconv
""" 

class DiscreteFunctionFromImage (DiscreteFunction):
    def __init__(self, path:str, coeffs:tuple=(0.299, 0.587, 0.114), x:int = 0, y:int = 0):
        import os

        if not round(sum(list(coeffs)), 2) == 1:
            raise DiscreteConvertionError(f"The sum of the coefficients for the conversion to grey level must be equal to 1 : {coeffs[0]} + {coeffs[1]} + {coeffs[2]} = {coeffs[0] + coeffs[1] + coeffs[2]} != 1")
        if not os.path.exists(path):
            raise DiscreteConvertionError(f"unknown access path : '{path}'")

        self.path: str = path
        self.coeffs: tuple = coeffs

        kernel: list[list[float]] = getKernelFromImage(Image.open(self.path), self.coeffs)

        super().__init__(kernel, x, y)


class GaussianDiscreteFunction (DiscreteFunction):
    def __init__(self, sigma: float, x: int = 0, y: int = 0):
        from math import pi

        self.sigma: float = sigma
        self.size: int = int(2*pi*sigma + 1)

        kernel = GaussianKernel(sigma, self.size)
        super().__init__(kernel, x, y)


class ComplexDiscreteFunction (DiscreteFunction):
    def __init__(self, kernel:list[list[complex]], x = 0, y = 0):
        super().__init__(kernel, x, y)
    
    def __mul__(self, value):
        if type(value) == float or type(value) == int:
            return ComplexDiscreteFunction([[self[i, j] * value for i in range(self.width)] for j in range(self.height)], x=self.x, y=self.y)
        elif type(value) == ComplexDiscreteFunction:
            
            mat=[]
            for j in range(self.height):
                mat.append([])
                for i in range(self.width):
                    if i < value.width and j < value.height:
                        mat[j].append(self[i,j]*value[i,j])
                    else:
                        mat[j].append(0)
            

            return ComplexDiscreteFunction(mat)
        else:
            raise TypeError(f"unsupported operand type(s) for '*' : 'DiscreteFunction' and '{ type(value).__name__}'")
    
    def __truediv__(self, value):
        if type(value) == float or type(value) == int:
            return ComplexDiscreteFunction([[self[i, j] / value for i in range(self.width)] for j in range(self.height)], x=self.x, y=self.y)
        elif type(value) == ComplexDiscreteFunction:
            
            mat=[]
            for j in range(self.height):
                mat.append([])
                for i in range(self.width):
                    if i < value.width and j < value.height:
                        if value[i,j]!=0:
                            mat[j].append(self[i,j]/value[i,j])
                        else:
                            mat[j].append(0)
                    else:
                        mat[j].append(0)
            

            return ComplexDiscreteFunction(mat)
        else:
            raise TypeError(f"unsupported operand type(s) for '*' : 'DiscreteFunction' and '{ type(value).__name__}'")

    def getModule(self, logarithmic:bool=True) -> DiscreteFunction:
        mat: list[list] = []

        for j in range(self.height):
            mat.append([])
            for i in range(self.width):
                if logarithmic: 
                    mat[j].append(log(1+abs(self[i,j]), 10))
                    """if self[i,j] != 0+0j: mat[j].append(log(abs(self[i,j]), 10))
                    else: mat[j].append(float("-inf"))"""
                else: mat[j].append(abs(self[i,j]))
        
        return DiscreteFunction(mat, 0, 0)

    def getArgument(self):
        mat=[]

        for j in range(self.height):
            mat.append([])
            for i in range(self.width):
                mat[j].append(arg(self[i,j]))
        
        return DiscreteFunction(mat, 0, 0)

    def show(self):
        #mat=ComplexDiscreteFunction(self.getRevolve().kernel)
        mat=self.getModule()
        mat.resizeAmplitude()
        mat=mat.getCentered()

        im=getImageFromDiscreteFunction(mat)
        im.show()
    
    def RadiusFilter(self, radiusFraction:float):
        radius=max(self.height, self.width)*radiusFraction/2

        hWidth=self.width/2
        hHeight=self.height/2

        for i in range(self.width):
            for j in range(self.height):
                if sqrt((i-hWidth)**2+(j-hHeight)**2) > radius:
                    self[i,j]=0
  
    def AmplitudeCutFilter(self, maxValueFraction, immunityRadius, logarithmic=True):
        if logarithmic: maxValue=log(1+abs(self[self.width//2, self.height//2]))*maxValueFraction
        else: maxValue=abs(self[self.width//2, self.height//2])*maxValueFraction

        hWidth=self.width/2
        hHeight=self.height/2

        for i in range(self.width):
            for j in range(self.height):
                if sqrt((i-hWidth)**2+(j-hHeight)**2) > immunityRadius:
                    if logarithmic:
                        if log(1+abs(self[i,j])) > maxValue:
                            self[i,j]=0
                    elif abs(self[i,j]) > maxValue:
                        self[i,j]=0
                
        mat : list[list] = []

        for j in range(self.height*2):
            mat.append([])
            for i in range(self.width*2):
                if j<self.height:
                    if i>=self.width:
                        mat[j].append(self[i-self.width, self.height-j])
                    else:
                        mat[j].append(self[self.width-i, self.height-j])
                else:
                    if i>=self.width:
                        mat[j].append(self[i-self.width, j-self.height])
                    else:
                        mat[j].append(self[self.width-i, j-self.height])
        
        return ComplexDiscreteFunction(mat)


class DiscreteConvertionError(Exception):
    pass



class ColorDiscreteFunction:
    def __init__(self, red_kernel, green_kernel, blue_kernel,x:int = 0, y:int = 0):
        self.R = red_kernel
        self.G = green_kernel
        self.B = blue_kernel
        self.width = red_kernel.width
        self.height = red_kernel.height
        self.x = x
        self.y = y

    def apply_to_all(self, func_name: str, *args, **kwargs):
        self.R = getattr(self.R, func_name)(*args, **kwargs)
        self.G = getattr(self.G, func_name)(*args, **kwargs)
        self.B = getattr(self.B, func_name)(*args, **kwargs)
        return self

    def show(self):
        image = getImageFromRGBFunctions(self.R, self.G, self.B)
        image.show()

    def save(self, filename: str):
        image = getImageFromRGBFunctions(self.R, self.G, self.B)
        image.save(filename)

class ColorDiscreteFunctionFromImage(ColorDiscreteFunction):
    def __init__(self, path: str, x: int = 0, y: int = 0):
        import os
        if not os.path.exists(path):
            raise DiscreteConvertionError(f"unknown access path : '{path}'")
        self.path = path

        red_kernel, green_kernel, blue_kernel = getRGBKernelsFromImage(Image.open(self.path))

        self.red_function = DiscreteFunction(red_kernel, x, y)
        self.green_function = DiscreteFunction(green_kernel, x, y)
        self.blue_function = DiscreteFunction(blue_kernel, x, y)

        super().__init__(self.red_function, self.green_function, self.blue_function, x, y)

