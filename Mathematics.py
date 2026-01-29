
import os

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
        if type(value) == int:
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
    
    def convolve(self, kernel: list[list[float]]):
        if type(kernel) != DiscreteFunction:
            raise TypeError

        g = DiscreteFunction(
            [[0]*self.width for _ in range(self.height)],
            x=self.x,
            y=self.y
        )
        for i in range(self.width):
            for j in range(self.height):
                g[i, j] = sum(
                    self[i - m, j - n] * kernel[m, n]
                    for m in range(kernel.width)
                    for n in range(kernel.height)
                )
        return g
    
class DiscretefunctionFromImage(DiscreteFunction):
    def __init__(self, path:str, coeffs:tuple=(0.299, 0.587, 0.114), x:int = 0, y:int = 0):

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

class DiscreteConvertionError(Exception):
    pass


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