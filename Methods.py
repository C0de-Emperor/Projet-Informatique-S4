from PIL import Image

def getImageFromDiscreteFunction(discreteFunction:DiscreteFunction) -> Image.Image:
    image=Image.new((discreteFunction.width, discreteFunction.height))

    for i in range(0, discreteFunction.width):
        for j in range(0, discreteFunction.height):
            image.putpixel((i,j), [discreteFunction[i,j]]*3)
    
    return image

def showImageFromDiscreteFunction(discreteFunction:DiscreteFunction):
    image=getImageFromDiscreteFunction(discreteFunction)

    image.show()

def saveImageFromDiscreteFunction(discreteFunction:DiscreteFunction, filename:str):
    image=getImageFromDiscreteFunction(discreteFunction)

    image.save(filename)