class discreteFunction:
    def __init__(self, kernel:list, x:int = 0, y:int = 0):
        self.kernel = kernel
        self.width = len(kernel[0])
        self.height = len(kernel)
        self.x = x
        self.y = y

    def __getitem__(self, item:tuple):
        #item[0] = x
        #item[1] = y
        if not (self.x <= item[0] < self.x + self.width):
            return 0
        if not (self.y <= item[1] < self.y + self.height):
            return 0
        return self.kernel[item[1] - self.y][item[0] - self.x]
    
    def __setitem__(self, item:tuple, value):
        #item[0] = x
        #item[1] = y
        if not (self.x <= item[0] < self.x + self.width):
            raise IndexError
        if not (self.y <= item[1] < self.y + self.height):
            raise IndexError
        self.kernel[item[1] - self.y][item[0] - self.x] = value

    def __add__(self, value):
        if type(value) == int:
            return discreteFunction([[self[i, j] + value for i in range(self.width)] for j in range(self.height)], x=self.x, y=self.y)
        elif type(value) == discreteFunction:
            xMax = max(self.width + self.x , value.width + value.x)
            yMax = max(self.height + self.y, value.height + value.y)
            xMin = min (self.x, value.x)
            yMin = min (self.y, value.y)
            return discreteFunction([[self[i, j] + value[i, j] for i in range(xMin, xMax)] for j in range(yMin, yMax)], x=xMin, y=yMin)
        else:
            raise TypeError(f"unsupported operand type(s) for '+' : 'DiscreteFunction' and '{ type(value).__name__}'")

    def __mul__(self, value):
        if type(value) == int:
            return discreteFunction([[self[i, j] * value for i in range(self.width)] for j in range(self.height)], x=self.x, y=self.y)
        elif type(value) == discreteFunction:
            xMax = min(self.width + self.x , value.width + value.x)
            yMax = min(self.height + self.y, value.height + value.y)
            xMin = max (self.x, value.x)
            yMin = max (self.y, value.y)
            return discreteFunction([[self[i, j] * value[i, j] for i in range(xMin, xMax)] for j in range(yMin, yMax)], x=xMin, y=yMin)
        else:
            raise TypeError(f"unsupported operand type(s) for '*' : 'DiscreteFunction' and '{ type(value).__name__}'")
        
    def __sub__(self, value):
        if type(value) == int:
            return discreteFunction([[self[i, j] - value for i in range(self.width)] for j in range(self.height)], x=self.x, y=self.y)
        elif type(value) == discreteFunction:
            xMax = max(self.width + self.x , value.width + value.x)
            yMax = max(self.height + self.y, value.height + value.y)
            xMin = min (self.x, value.x)
            yMin = min (self.y, value.y)
            return discreteFunction([[self[i, j] - value[i, j] for i in range(xMin, xMax)] for j in range(yMin, yMax)], x=xMin, y=yMin)
        else:
            raise TypeError(f"unsupported operand type(s) for '-' : 'DiscreteFunction' and '{ type(value).__name__}'")
        
    def __eq__(self, value):
        if type(value) != discreteFunction:
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
    
    def convolve(self, kernel):
        if type(kernel) != discreteFunction:
            raise TypeError

        g = discreteFunction(
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

from PIL import Image

def getNiveauxGris(filepath:str, coeffs:tuple=(0.299, 0.587, 0.114), showImage:bool=False):
    # coeffs[0], coeffs[1], coeffs[3] sont les coefficients de la combinaison linÃ©aire respectivement de rouge, vert et bleu
    assert round(sum(list(coeffs)),2)==1

    image=Image.open(filepath)
    imageMat=[]
    for j in range(image.height):
        imageMat.append([])
        for i in range(image.width):
            pixelColors=image.getpixel((i,j))
            imageMat[j].append(round(sum([coeffs[k]*pixelColors[k] for k in range(3)])))

    if showImage:
        image2=Image.new("RGB", image.size)
        for j in range(image2.height):
            for i in range(image2.width):
                image2.putpixel((i,j), tuple([imageMat[j][i]]*3))
        image2.show()

    return imageMat

print(getNiveauxGris(input("chemin de l'image ? : "), (0.299, 0.587, 0.114), True))


f = discreteFunction(
    [
        [255, 40, 30, 20, 10],
        [39, 38, 138, 130, 12],
        [7, 210, 186, 1, 1],
        [200, 210, 186, 1, 1],
        [100, 210, 186, 1, 1]
    ],
    x = 0,
    y = 0
)

h = discreteFunction(
    [
        [1, 3, 1],
        [2, 3, 2],
        [1, 2, 1]
    ],
    x = 0,
    y = 0
)

#print(f.convolve(h).kernel)
