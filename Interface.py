import customtkinter as ctk
import tkinter as tk
from PIL import Image, UnidentifiedImageError
import inspect
from functools import partial
from multiprocessing import Pool, pool
from pyperclipimg import copy, paste

from Mathematics import *
from MathematicsMethods import *
from ImageMethods import *
from Noising import *
from Analysis import *

ERROR_SHOW_TIME=5000

nextAvailableId=1
tabs=[]


class Tab:
    def __init__(self, name:str, parent=None):
        global nextAvailableId

        self.alive=True
        self.frequencyDomain=False
        self.trueSize=None

        self.id=nextAvailableId
        nextAvailableId+=1

        self.image:Image.Image=None
        self.discreteFunction:DiscreteFunction=None

        self.infos={}
        
        self.name=name
        self.parentTab=parent
        if parent==None: 
            self.parentFrame=tabsFrame
            tabs.append(self)
        else: 
            self.parentFrame=parent.tabFrame
            parent.childrenTabs.append(self)

        self.childrenTabs=[]

        self.createTabElement()
    
    def createTabElement(self):
        self.tabFrame=ctk.CTkFrame(self.parentFrame)
        self.tabFrame.pack(fill="x", padx=(20*(self.parentFrame!=tabsFrame),0))

        self.tabButton=ctk.CTkButton(self.tabFrame, text=self.name, command=lambda: self.showSelf(True, True, True))
        self.tabButton.pack(fill="x")
    
    def showSelf(self, upImage:bool=False, upInfos:bool=False, clicked:bool=False):
        global currentTab

        if currentTab != None: currentTab.tabButton.configure(fg_color=("#3B8ED0", "#1F6AA5"))
        self.tabButton.configure(fg_color=("#1F6AA5", "#19486D"))
        
        currentTab=self

        if upImage: self.updateImage()
        if upInfos:imageInfosPanel.updateInfos(self.infos)
        if clicked:
            imageProcessingPanel.destroyButtons()
            imageProcessingPanel.setButtons()

        if self.discreteFunction == None or not isinstance(self.discreteFunction, DiscreteFunction): imageProcessingPanel.destroyButtons()

        fileEditingPanel.updateName(self.name)
    
    def updateImage(self):
        global imageContainer

        if isinstance(self.image, Image.Image):
            scalingFactor=min(imageContainer.winfo_width()/self.image.width, imageContainer.winfo_height()/self.image.height)/window._get_window_scaling()

            image=ctk.CTkImage(self.image, self.image, (self.image.width*scalingFactor, self.image.height*scalingFactor))
            imageContainer.configure(image=image)
            imageContainer.configure(text="")
        else:
            imageContainer.pack_forget()

            imageContainer=ctk.CTkLabel(middleContainer, text="Image is loading...")
            imageContainer.pack(fill="both", expand=True)
    
    def getInfos(self):
        self.infos={}

        if isinstance(self.image, Image.Image):
            self.infos["Width"]=self.image.width
            self.infos["Height"]=self.image.height
            self.infos["Number of pixels"]=self.image.width*self.image.height
        
        if isinstance(self.discreteFunction, DiscreteFunction):
            self.infos["Local Variance"]=mainPool.apply_async(getInfoForCallback, (self.discreteFunction, "Local Variance"), callback=self.changeInfo)
            self.infos["Gradient Energy"]=mainPool.apply_async(getInfoForCallback, (self.discreteFunction, "Gradient Energy"), callback=self.changeInfo)
            self.infos["High Frequency Ratio"]=mainPool.apply_async(getInfoForCallback, (self.discreteFunction, "High Frequency Ratio"), callback=self.changeInfo)
    
    def changeInfo(self, element):
        if not self.alive: return

        self.infos[element[1]]=element[0]

        if currentTab==self:
            self.showSelf(upInfos=True)
    
    def getTrueSize(self):
        if self.trueSize == None:
            return self.parentTab.getTrueSize()
        return self.trueSize

class TabFromImage(Tab):
    def __init__(self, name:str, image:Image.Image, parent:Tab=None):
        super().__init__(name, parent)
        
        self.image=mainPool.apply_async(getGrayScaleImage, (image, (0.299, 0.587, 0.114)), callback=self.changeImage)
    
    def changeImage(self, element):
        if not self.alive: return

        self.image=element
        self.trueSize=self.image.size

        self.discreteFunction=mainPool.apply_async(getKernelFromImage, (self.image, (0.299, 0.587, 0.114)), callback=self.changeDiscreteFunction)

        self.getInfos()
        if currentTab == self:
            self.showSelf(upInfos=True, upImage=True)
    
    def changeDiscreteFunction(self, element):
        if not self.alive: return 

        self.discreteFunction=DiscreteFunction(element)

        self.getInfos()
        
        window.after(0, imageProcessingPanel.setButtons)

class TabFromDiscreteFunction(Tab):
    def __init__(self, name:str, discreteFunction:DiscreteFunction, parent:Tab=None, frequencyDomain:bool=False):
        super().__init__(name[1:], parent)

        self.frequencyDomain=frequencyDomain
        self.discreteFunction=discreteFunction
        self.image=mainPool.apply_async(TabFromDiscreteFunction.getImageOrModule, (self.discreteFunction, self.frequencyDomain), callback=self.changeImage)
        self.getInfos()
        
        #imageProcessingPanel.setButtons()
    
    def getImageOrModule(discreteFunction:DiscreteFunction, frequencyDomain:DiscreteFunction):
        if frequencyDomain:
            discreteFunction=discreteFunction.getModule()
            discreteFunction.resizeAmplitude()
        
        return getImageFromDiscreteFunction(discreteFunction)
    
    def changeImage(self, element):
        if not self.alive: return

        self.image=element

        self.getInfos()
        if currentTab==self:
            self.showSelf(upImage=True)

class TabFromFunction(Tab):
    def __init__(self, name:str, function, args, parent:Tab=None):
        super().__init__("*"+name, parent)

        self.function=function
        mainPool.apply_async(TabFromFunction.applyFunction, (parent.discreteFunction, function, args), callback=self.changeDiscreteFunction)
    
    def changeDiscreteFunction(self, element):
        if not self.alive: return 

        window.after(0, self.tabFrame.pack_forget())

        if self.function==ComplexDiscreteFunctionIFFT2:
            trueSize=self.getTrueSize()
            kernel=element.kernel[:trueSize[1]]
            kernel=[[element[i,j] for i in range(trueSize[0])] for j in range(len(kernel))]
            element=DiscreteFunction(kernel)

            tab=TabFromDiscreteFunction(self.name, element, self.parentTab, not self.parentTab.frequencyDomain)
        elif self.function==DiscreteFunctionFFT2:
            tab=TabFromDiscreteFunction(self.name, element, self.parentTab, not self.parentTab.frequencyDomain)
        else:
            tab=TabFromDiscreteFunction(self.name, element, self.parentTab, self.parentTab.frequencyDomain)

        if currentTab==self:
            tab.showSelf(upImage=True, upInfos=True, clicked=True)

        self.parentTab.childrenTabs.remove(self)
    
    def applyFunction(discreteFunction, function, args):

        newArgs=dict(zip(inspect.signature(function).parameters, [discreteFunction]+args))

        try: a=function(**newArgs)
        except Exception as e: print(e)

        if a==None: return discreteFunction
        else: return a


class ImageInfosPanel:
    def __init__(self):
        self.frame=ctk.CTkFrame(rightContainer)
        self.frame.pack(side="top", fill="x")
        
        ctk.CTkLabel(self.frame, text="Image Characteristics :", font=titleFont).pack()
        
        self.infos={"Width":0, "Height":0, "Number of pixels":0, "Local Variance":0, "Gradient Energy":0, "High Frequency Ratio":4}

        self.setInfos()
    
    def setInfos(self):
        self.infosLabels={}

        for k in self.infos.keys():
            label=ctk.CTkLabel(self.frame, text=k+": -")
            label.pack()
            self.infosLabels[k]=label

    def updateInfos(self, newInfos:dict):
        for key, value in self.infosLabels.items():
            if key in newInfos.keys() and (isinstance(newInfos[key], int) or isinstance(newInfos[key], float)):
                value.configure(text=key+": "+str(round(newInfos[key], self.infos[key])))
            else:
                value.configure(text=key+": -")

class ImageProcessingPanel:
    def __init__(self):
        self.frame=ctk.CTkFrame(rightContainer)
        self.frame.pack(side="top", fill="x", pady=(5,0))

        self.spatialFunctions={"Bruitage poivre et sel": (saltAndPaperNoising, "probability"),
                                "Bruitage aléatoire": (randomNoising, "minAdd", "maxAdd"),
                                "Filtre médian": (DiscreteFunction.medianFilter, "radius"),
                                "Filtre adaptatif": (DiscreteFunctionAdaptativeFilter, "kernel size", "max diff"),
                                "Filtre bilatéral": (DiscreteFunctionBilateralFilter, "kernel size", "variance de la normale")}
        
        self.frequencyFunctions={"Filtre d'amplitude": (ComplexDiscreteFunction.AmplitudeCutFilter, "maxAmp fraction"),
                                "Filtre passe bas": (ComplexDiscreteFunction.RadiusFilter, "radius fraction")}
    
    def destroyButtons(self):
        for widget in self.frame.winfo_children():
            widget.destroy()
    
    def setButtons(self):
        self.buttons={}

        ctk.CTkLabel(self.frame, text="Image Processing Actions :", font=titleFont).pack(padx=10)

        if currentTab.frequencyDomain:
            self.functions=self.frequencyFunctions
            self.functions["IFFT2"]=(ComplexDiscreteFunctionIFFT2,)
        else:
            self.functions=self.spatialFunctions
            self.functions["FFT2"]=(DiscreteFunctionFFT2,)

        for key, value in self.functions.items():
            elementsList=[]

            buttonFrame=ctk.CTkFrame(self.frame)
            buttonFrame.pack(pady=10)
            elementsList.append(buttonFrame)

            button=ctk.CTkButton(buttonFrame, text=key, command=partial(self.buttonPressed, key))
            button.grid(column=0, row=0, padx=5)
            elementsList.append(button)

            elementsList.append([])
            for n in range(1, len(value)):
                input=ctk.CTkEntry(buttonFrame, placeholder_text=value[n], width=len(value[n])*10)
                input.grid(column=n, row=0, padx=5)
                elementsList[2].append(input)
            
            self.buttons[key]=elementsList
    
    def buttonPressed(self, key):
        if not isinstance(currentTab.discreteFunction, DiscreteFunction):
            print("ARGGG")
            return

        args=[float(k.get()) for k in self.buttons[key][2]]

        name=key+"; "+"; ".join([self.functions[key][k]+":"+str(args[k-1]) for k in range(1, len(self.functions[key]))])

        tab=TabFromFunction(name, self.functions[key][0], args, currentTab)
        tab.showSelf(upImage=True, upInfos=True)

class FileEditingPanel:
    def __init__(self):
        self.frame=ctk.CTkFrame(middleContainer, height=50)
        self.frame.pack(side="bottom", fill="x")
        self.frame.rowconfigure(0, weight=1)
        self.frame.columnconfigure(0, weight=2)

        self.fileNameTextBox=ctk.CTkEntry(self.frame)
        self.fileNameTextBox.grid(row=0, column=0, sticky="we")

        saveImageImage=ctk.CTkImage(Image.open("InterfaceData/renameL.png"), Image.open("InterfaceData/renameD.png"))
        self.renameButton=ctk.CTkButton(self.frame, image=saveImageImage, text="", width=saveImageImage._size[0], height=saveImageImage._size[1], command=self.renameImage)
        self.renameButton.grid(row=0, column=1, sticky="e")

        saveImageImage=ctk.CTkImage(Image.open("InterfaceData/saveL.png"), Image.open("InterfaceData/saveD.png"))
        self.saveButton=ctk.CTkButton(self.frame, image=saveImageImage, text="", width=saveImageImage._size[0], height=saveImageImage._size[1], command=self.saveImage)
        self.saveButton.grid(row=0, column=2, sticky="e")

        copyImageImage=ctk.CTkImage(Image.open("InterfaceData/copyL.png"), Image.open("InterfaceData/copyD.png"))
        self.copyImageButton=ctk.CTkButton(self.frame, image=copyImageImage, text="", width=copyImageImage._size[0], height=copyImageImage._size[1], command=self.copyImage)
        self.copyImageButton.grid(row=0, column=3, sticky="e")

        deleteImageImage=ctk.CTkImage(Image.open("InterfaceData/deleteL.png"), Image.open("InterfaceData/deleteD.png"))
        self.deleteImageButton=ctk.CTkButton(self.frame, image=deleteImageImage, text="", width=deleteImageImage._size[0], height=deleteImageImage._size[1], command=self.deleteImage)
        self.deleteImageButton.grid(row=0, column=4, sticky="e")
    
    def updateName(self, newName):
        self.fileNameTextBox.delete(0, ctk.END)
        self.fileNameTextBox.insert(0, newName)
    
    def renameImage(self):
        currentTab.name=self.fileNameTextBox.get()
        currentTab.tabButton.configure(text=self.fileNameTextBox.get())
    
    def saveImage(self):
        filepath=tk.filedialog.asksaveasfilename(title = "Select file", filetypes=(("png file", "*.png"),))
        if filepath=="": return

        currentTab.image.save(filepath+".png")

    def copyImage(self, event=None):
        copy(currentTab.image)
    
    def deleteImage(self):
        global currentTab, imageContainer

        if currentTab==None: return
        
        currentTab.alive=False
        currentTab.tabFrame.pack_forget()
        
        if currentTab.parentTab == None:
            tabs.remove(currentTab)
            if len(tabs)!=0: 
                currentTab=tabs[0]
                currentTab.showSelf(upImage=True, upInfos=True, clicked=True)
            else: 
                currentTab=None

                imageContainer.pack_forget()
                imageContainer=ctk.CTkLabel(middleContainer, text="Image visualisation window")
                imageContainer.pack(side="top", fill="both", expand=True)

                imageInfosPanel.updateInfos({})
                imageProcessingPanel.destroyButtons()
                self.updateName("")
        else:
            currentTab.parentTab.childrenTabs.remove(currentTab)
            if len(currentTab.parentTab.childrenTabs)!=0:
                currentTab=currentTab.parentTab.childrenTabs[0]
                currentTab.showSelf(upImage=True, upInfos=True, clicked=True)
            else:
                currentTab=currentTab.parentTab
                currentTab.showSelf(upImage=True, upInfos=True, clicked=True)


def DiscreteFunctionBilateralFilter(discreteFunction:DiscreteFunction, kernelSize:float, sigma_r:float):
    gaussianDiscreteFunction=GaussianDiscreteFunction(kernelSize)

    return discreteFunction.bilateralFilter(gaussianDiscreteFunction, sigma_r)

def DiscreteFunctionAdaptativeFilter(discreteFunction:DiscreteFunction, kernelSize:float, diff:float):
    gaussianDiscreteFunction=GaussianDiscreteFunction(kernelSize)

    return discreteFunction.adaptativeGaussianConvolution(gaussianDiscreteFunction, diff)

def ComplexDiscreteFunctionIFFT2(discreteFunction:ComplexDiscreteFunction):
    ifft2Kernel=IFFT2(discreteFunction.kernel, 2)
    
    IFFT2DiscreteFunction=DiscreteFunction(ifft2Kernel)
    return IFFT2DiscreteFunction

def DiscreteFunctionFFT2(discreteFunction:DiscreteFunction, ) -> DiscreteFunction:
    #if discreteFunction.width*discreteFunction.height > 512**2: fft2Kernel=FFT2Boost(discreteFunction.kernel, pool=mainPool)
    #else: fft2Kernel=FFT2(discreteFunction.kernel)
    fft2Kernel=FFT2(discreteFunction.kernel)

    FFT2DiscreteFunction=ComplexDiscreteFunction(fft2Kernel)

    return FFT2DiscreteFunction

def importImageFromClipboard(event):
    image=paste()
    if image==None: return

    tab=TabFromImage("image", image)
    tab.showSelf(upImage=True, upInfos=True)

def importNewImage():
    filepath=tk.filedialog.askopenfilename(title="Select File", filetypes=(("all files","*.*"),("png files","*.png"),("jpeg files","*.jpg")))
    if filepath=="": return
    filename=os.path.basename(filepath)

    try:
        image=Image.open(filepath)
    except UnidentifiedImageError as e:
        #raise e
        addErrorMessage(f"File format .{filename.split(".")[1]} not supported")
        return

    tab=TabFromImage(filename.split(".")[:-1], image)
    tab.showSelf(upImage=True, upInfos=True)

def getInfoForCallback(discreteFunction, infoName):
    a={"Local Variance":LocalVariance, "Gradient Energy":GradientEnergy, "High Frequency Ratio":HighFrequencyRatio}
    return a[infoName](discreteFunction), infoName

def addErrorMessage(message:str):
    errorMessageTextboxValue=errorMessageLabel.cget("text")
    if errorMessageTextboxValue=="": newErrorMessage=message
    else: newErrorMessage=errorMessageTextboxValue+"\n"+message

    errorMessageLabel.configure(text=newErrorMessage)
    errorMessageLabel.pack(fill="x", side="top")

    window.after(ERROR_SHOW_TIME, removeErrorMessage)

def removeErrorMessage():
    errorMessageTextboxValue=errorMessageLabel.cget("text")
    errorMessageLines=errorMessageTextboxValue.split("\n")
    errorMessageLabel.configure(text="\n".join(errorMessageLines[1:]))

    if len(errorMessageLines)<=1:
        errorMessageLabel.pack_forget()




currentTab:Tab=None

window=ctk.CTk()
window.title="Projet Maths-Info S4"
window.geometry("1000x600")
window.bind("<Control-v>", importImageFromClipboard)

titleFont=ctk.CTkFont("font1", 13, "bold", underline=True)


panedWindow=tk.PanedWindow(window, orient="horizontal", background="black")
panedWindow.pack(fill="both", expand=True)


errorMessageLabel=ctk.CTkLabel(window, text="")


leftContainer=ctk.CTkFrame(panedWindow)
panedWindow.add(leftContainer, minsize=300, stretch="never")

tabsFrame=ctk.CTkFrame(leftContainer)
tabsFrame.pack(fill="both", expand=True)

importImageButton=ctk.CTkButton(leftContainer, text="Import Image", command=importNewImage)
importImageButton.pack(side="bottom", fill="x")


middleContainer=ctk.CTkFrame(panedWindow)
panedWindow.add(middleContainer, minsize=200, stretch="always")

imageContainer=ctk.CTkLabel(middleContainer, text="Image visualisation window")
imageContainer.pack(side="top", fill="both", expand=True)

fileEditingPanel=FileEditingPanel()
window.bind("<Control-c>", fileEditingPanel.copyImage)


rightContainer=ctk.CTkFrame(panedWindow)
panedWindow.add(rightContainer, minsize=300, stretch="never")

imageInfosPanel=ImageInfosPanel()

imageProcessingPanel=ImageProcessingPanel()

if __name__=="__main__":
    mainPool=Pool()

    window.mainloop()