from math import e, pi, atan2, log

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
    
    def getModule(self, logarithmic:bool=True):
        mat=[]

        for j in range(self.height):
            mat.append([])
            for i in range(self.width):
                if logarithmic: 
                    if self[i,j] != 0+0j: mat[j].append(log(abs(self[i,j]), 10))
                    else: mat[j].append(float("-inf"))
                else: mat[j].append(abs(self[i,j]))
        
        return DiscreteFunction(mat, 0, 0)

    def getArgument(self):
        mat=[]

        for j in range(self.height):
            mat.append([])
            for i in range(self.width):
                mat[j].append(atan2(self[i,j].real, self[i,j].imag))
        
        return DiscreteFunction(mat, 0, 0)

    def resizeAmplitudeDiscreteFunction(self, minValue:int=0, maxValue:int=255):
        minV=float("inf")
        maxV=float("-inf")

        for i in range(self.width):
            for j in range(self.height):
                if self[i,j]>maxV: maxV=self[i,j]
                if self[i,j]<minV: minV=self[i,j]

        for i in range(self.width):
            for j in range(self.height):
                self[i,j]=(self[i,j]-minV)*(maxValue-minValue)/(maxV-minV)+minValue
    
    def medianFilter(self, radius: int = 1):
        newKernel = [[0]*self.width for _ in range(self.height)]

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

    def getStandardDeviation(self):
        from math import sqrt

        N = self.width * self.height
        mean = sum(self[i, j] for j in range(self.height) for i in range(self.width)) / N

        variance = sum(
            (self[i, j] - mean)**2
            for j in range(self.height)
            for i in range(self.width)
        ) / N

        return sqrt(variance)

    def apply(self, func, *args, **kwargs):
        func(self, *args, **kwargs)
        return self



class DiscretefunctionFromImage (DiscreteFunction):
    def __init__(self, path:str, coeffs:tuple=(0.299, 0.587, 0.114), x:int = 0, y:int = 0):
        import os

        if not round(sum(list(coeffs)), 2) == 1:
            raise DiscreteConvertionError(f"The sum of the coefficients for the conversion to grey level must be equal to 1 : {coeffs[0]} + {coeffs[1]} + {coeffs[2]} = {coeffs[0] + coeffs[1] + coeffs[2]} != 1")
        if not os.path.exists(path):
            raise DiscreteConvertionError(f"unknown access path : '{path}'")

        self.path: str = path
        self.coeffs: tuple = coeffs

        kernel: list[list[float]] = DiscretefunctionFromImage._GetGrayScaleKernel(path, coeffs)

        super().__init__(kernel, x, y)

    @staticmethod
    def _GetGrayScaleKernel(filepath:str, coeffs:tuple=(0.299, 0.587, 0.114), showImage:bool=False) -> list[list[float]]:
        from PIL import Image
        # coeffs[0], coeffs[1], coeffs[3] sont les coefficients de la combinaison linÃ©aire respectivement de rouge, vert et bleu

        image = Image.open(filepath)
        imageKernel = []
        for j in range(image.height):
            imageKernel.append([])
            for i in range(image.width):
                pixelColors = image.getpixel((i,j))
                imageKernel[j].append(round(sum([coeffs[k]*pixelColors[k] for k in range(3)])))
        
        if showImage:
            image2 = Image.new("RGB", image.size)
            for j in range(image2.height):
                for i in range(image2.width):
                    image2.putpixel((i,j), tuple([imageKernel[j][i]]*3))
            image2.show()

        return imageKernel


class GaussianDiscreteFunction (DiscreteFunction):
    def __init__(self, sigma: float, x: int = 0, y: int = 0):
        from math import pi

        self.sigma: float = sigma
        self.size: int = int(2*pi*sigma + 1)

        kernel = GaussianDiscreteFunction._GetGaussianKernel(sigma, self.size)
        super().__init__(kernel, x, y)

    @staticmethod
    def _GetGaussianKernel(sigma: float, size: int) -> list[list[float]]:
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


class FrequencyDiscreteFunction (DiscreteFunction):
    def __init__(self, kernel:list[list[complex]], x = 0, y = 0):
        super().__init__(self.revolve(kernel), x, y)
    
    def revolve(self, kernel:list[list[complex]]):
        width=len(kernel[0])
        height=len(kernel)

        mat : list[list] = []
        for j in range(height*2):
            mat.append([])
            for i in range(width*2):
                if j<height:
                    if i>=width:
                        mat[j].append(kernel[-j][i-width])
                    else:
                        mat[j].append(kernel[-j][-i])
                else:
                    if i>=width:
                        mat[j].append(kernel[j-height][i-width])
                    else:
                        mat[j].append(kernel[j-height][-i])
        
        return mat


class DiscreteConvertionError(Exception):
    pass


def FourierTransform(discreteFunction:DiscreteFunction, rayonMax:int=-1) -> DiscreteFunction:
    mat=[]
    for q in range(discreteFunction.height):
        mat.append([])
        print(round(q/discreteFunction.height*100, 1), "%")
        for p in range(discreteFunction.width):
            value=0
            if rayonMax>=0:
                for m in range(p-rayonMax, p+rayonMax+1):
                    for n in range(q-rayonMax, q+rayonMax+1):
                        theta=-2*pi*(p*(rayonMax+m)/(2*rayonMax+1)+q*(rayonMax+n)/(2*rayonMax+1))
                        value+=discreteFunction[m,n]*e**(theta*1j)
            else:
                for m in range(discreteFunction.width):
                    for n in range(discreteFunction.height):
                        theta=-2*pi*(p*m/discreteFunction.width+q*n/discreteFunction.height)
                        value+=discreteFunction[m,n]*e**(theta*1j)
            
            mat[q].append(value)
    
    return FrequencyDiscreteFunction(mat, 0, 0)



"""
def getNiveauxGris(filepath:str, coeffs:tuple=(0.299, 0.587, 0.114), showImage:bool=False):
    # coeffs[0], coeffs[1], coeffs[3] sont les coefficients de la combinaison linÃ©aire respectivement de rouge, vert et bleu
    assert round(sum(list(coeffs)), 2) == 1
    assert os.path.exists(filepath)

    image = PIL.Image.open(filepath)
    imageMat = []

    for j in range(image.height):
        imageMat.append([])
        for i in range(image.width):
            pixelColors=image.getpixel((i,j))
            imageMat[j].append(round(sum([coeffs[k]*pixelColors[k] for k in range(3)])))

    if showImage:
        image2 = PIL.Image.new("RGB", image.size)
        for j in range(image2.height):
            for i in range(image2.width):
                image2.putpixel((i,j), tuple([imageMat[j][i]]*3))
        image2.show()

    return imageMat
"""