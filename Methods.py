from PIL import Image
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Mathematics import DiscreteFunction

def getImageFromDiscreteFunction(discreteFunction:"DiscreteFunction") -> Image.Image:
    image=Image.new("RGB", (discreteFunction.width, discreteFunction.height))

    for i in range(0, discreteFunction.width):
        for j in range(0, discreteFunction.height):
            image.putpixel((i,j), tuple([int(discreteFunction[i,j])]*3))
    
    return image

def showImageFromDiscreteFunction(discreteFunction:"DiscreteFunction"):
    image=getImageFromDiscreteFunction(discreteFunction)

    image.show()

def saveImageFromDiscreteFunction(discreteFunction:"DiscreteFunction", filename:str):
    image=getImageFromDiscreteFunction(discreteFunction)

    image.save(filename)