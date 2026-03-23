from PIL import Image
from typing import TYPE_CHECKING
import os
from math import isnan

if TYPE_CHECKING:
    from Mathematics import DiscreteFunction

def getKernelFromImage(image:Image.Image, coeffs:tuple=(0.299, 0.587, 0.114)) -> list[list[float]]:
        # coeffs[0], coeffs[1], coeffs[3] sont les coefficients de la combinaison lineaire respectivement de rouge, vert et bleu

        imageKernel = []
        if type(image.getpixel((0,0)))==int:
            for j in range(image.height):
                imageKernel.append([])
                for i in range(image.width):
                    pixelColors = image.getpixel((i,j))
                    imageKernel[j].append(pixelColors)
        else:
            for j in range(image.height):
                imageKernel.append([])
                for i in range(image.width):
                    pixelColors = image.getpixel((i,j))
                    imageKernel[j].append(round(sum([coeffs[k]*pixelColors[k] for k in range(3)])))

        return imageKernel

def getGrayScaleImage(image:Image.Image, coeffs:tuple=(0.299, 0.587, 0.114)) -> Image.Image:
    image2=Image.new("RGB", image.size)

    for j in range(image.height):
        for i in range(image.width):
            pixelColors=image.getpixel((i,j))
            grayLevel=round(sum([coeffs[k]*pixelColors[k] for k in range(3)]))
            image2.putpixel((i,j), (grayLevel, grayLevel, grayLevel))

    return image2

def getImageFromDiscreteFunction(discreteFunction:"DiscreteFunction") -> Image.Image:
    image=Image.new("RGB", (discreteFunction.width, discreteFunction.height))

    for i in range(0, discreteFunction.width):
        for j in range(0, discreteFunction.height):
            if discreteFunction[i,j]==None or isnan(discreteFunction[i,j]) or discreteFunction[i,j]==float("-inf") or discreteFunction[i,j]==float("inf"): image.putpixel((i,j), (255,0,0))
            else: 
                a=int(discreteFunction[i,j])
                if a>=255: a=255
                if a<=0: a=0
                image.putpixel((i,j), tuple([a]*3))
    
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


def getRGBKernelsFromImage(image):
    image = image.convert("RGB")
    red_kernel, green_kernel, blue_kernel = [], [], []
    for j in range(image.height):
        red_kernel.append([])
        green_kernel.append([])
        blue_kernel.append([])
        for i in range(image.width):
            r, g, b = image.getpixel((i, j))
            red_kernel[j].append(r)
            green_kernel[j].append(g)
            blue_kernel[j].append(b)
    return red_kernel, green_kernel, blue_kernel

def getImageFromRGBFunctions(red_function, green_function, blue_function):
    image = Image.new("RGB", (red_function.width, red_function.height))
    for i in range(red_function.width):
        for j in range(red_function.height):
            r = max(0, min(255, int(abs(red_function[i, j]))))
            g = max(0, min(255, int(abs(green_function[i, j]))))
            b = max(0, min(255, int(abs(blue_function[i, j]))))
            image.putpixel((i, j), (r, g, b))
    return image

