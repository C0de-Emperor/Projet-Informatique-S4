from PIL import Image
from typing import TYPE_CHECKING
import os
from math import isnan

if TYPE_CHECKING:
    from Mathematics import DiscreteFunction

def GetGrayScaleImage(filepath:str, coeffs:tuple=(0.299, 0.587, 0.114)) -> list[list[float]]:
        from PIL import Image
        # coeffs[0], coeffs[1], coeffs[3] sont les coefficients de la combinaison lineaire respectivement de rouge, vert et bleu

        image = Image.open(filepath)
        imageKernel = []

        for j in range(image.height):
            imageKernel.append([])
            for i in range(image.width):
                pixelColors = image.getpixel((i,j))
                imageKernel[j].append(round(sum([coeffs[k]*pixelColors[k] for k in range(3)])))

        return imageKernel

def getImageFromDiscreteFunction(discreteFunction:"DiscreteFunction") -> Image.Image:
    image=Image.new("RGB", (discreteFunction.width, discreteFunction.height))

    for i in range(0, discreteFunction.width):
        for j in range(0, discreteFunction.height):
            if discreteFunction[i,j]==None or isnan(discreteFunction[i,j]) or discreteFunction[i,j]==float("-inf") or discreteFunction[i,j]==float("inf"): image.putpixel((i,j), (255,0,0))
            else: image.putpixel((i,j), tuple([int(abs(discreteFunction[i,j]))]*3))
    
    return image

def saveImageFromDiscreteFunction(discreteFunction:"DiscreteFunction", filename:str):
    image=getImageFromDiscreteFunction(discreteFunction)

    image.save(filename)

def saveDiscreteFunction(discreteFunction:"DiscreteFunction", filename:str):
    with open(filename, "w") as f:
        f.write(str(discreteFunction.x)+"\n")
        f.write(str(discreteFunction.y)+"\n")
        for j in range(discreteFunction.height):
            for i in range(discreteFunction.width):
                f.write(str(discreteFunction[i,j]))
                if i!=discreteFunction.width-1: f.write(",")
            f.write("\n")

def loadDiscreteFunction(filename:str):
    if not os.path.exists(filename):
        raise Exception

    with open(filename, "r") as f:
        mat=f.readlines()
    
    x=int(mat[0])
    y=int(mat[1])
    mat=[mat[k].rstrip().split(",") for k in range(2, len(mat))]

    kernel=[]
    for j in range(len(mat)):
        kernel.append([])
        for i in range(len(mat[0])):
            kernel[j].append(complex(mat[j][i]))
    
    return DiscreteFunction(kernel, x, y)