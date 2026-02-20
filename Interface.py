import customtkinter as ctk
import tkinter as tk
from PIL import Image, UnidentifiedImageError
import inspect
from functools import partial
from multiprocessing import Pool

from Mathematics import *
from MathematicsMethods import *
from ImageMethods import *
from Noising import *
from Analysis import *

ERROR_SHOW_TIME=5000

nextAvailableId=1
tabs=[]

currentTab=None

class Tab:
    def __init__(self, name:str, parent:ctk.CTkFrame):
        global nextAvailableId

        self.id=nextAvailableId
        nextAvailableId+=1

        self.image=None
        self.discreteFunction=None

        self.infos={}
        
        self.name=name
        self.parent=parent

        self.createTabElement()

        tabs.append(self)
    
    def createTabElement(self):
        self.tabFrame=ctk.CTkFrame(self.parent)
        self.tabFrame.pack(fill="x", padx=(20*(self.parent!=tabsFrame),0))

        self.tabButton=ctk.CTkButton(self.tabFrame, text=self.name, command=lambda: self.showSelf(True, True, True))
        self.tabButton.pack(fill="x")
    
    def showSelf(self, upImage:bool=False, upInfos:bool=False, clicked:bool=False):
        global currentTab

        if currentTab != None: currentTab.tabButton.configure(fg_color=("#3B8ED0", "#1F6AA5"))
        self.tabButton.configure(fg_color=("#1F6AA5", "#3B8ED0"))

        if upImage: self.updateImage()
        if upInfos:imageInfosPanel.updateInfos(self.infos)
        if clicked:
            imageProcessingPanel.destroyButtons()
            imageProcessingPanel.setButtons()

        if self.discreteFunction == None or not isinstance(self.discreteFunction, DiscreteFunction): imageProcessingPanel.destroyButtons()

        currentTab=self
    
    def updateImage(self):
        global imageContainer

        if isinstance(self.image, Image.Image):
            scalingFactor=min(imageContainer.winfo_width()/self.image.width, imageContainer.winfo_height()/self.image.height)

            imageContainer.pack_forget()

            image=ctk.CTkImage(self.image, self.image, (self.image.width*scalingFactor, self.image.height*scalingFactor))
            imageContainer=ctk.CTkLabel(middleContainer, image=image, text="")

            imageContainer.pack(fill="both", expand=True)
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
            self.infos["Local Variance"]=pool.apply_async(getInfoForCallback, (self.discreteFunction, "Local Variance"), callback=self.changeInfo)
            self.infos["Gradient Energy"]=pool.apply_async(getInfoForCallback, (self.discreteFunction, "Gradient Energy"), callback=self.changeInfo)
            self.infos["High Frequency Ratio"]=pool.apply_async(getInfoForCallback, (self.discreteFunction, "High Frequency Ratio"), callback=self.changeInfo)
    
    def changeInfo(self, element):
        self.infos[element[1]]=element[0]

        if currentTab==self:
            self.showSelf(upInfos=True)

class TabFromImage(Tab):
    def __init__(self, name:str, parent:ctk.CTkFrame, image:Image.Image):
        super().__init__(name, parent)
        
        self.image=pool.apply_async(getGrayScaleImage, (image, (0.299, 0.587, 0.114)), callback=self.changeImage)
    
    def changeImage(self, element):
        self.image=element

        self.discreteFunction=pool.apply_async(getKernelFromImage, (self.image, (0.299, 0.587, 0.114)), callback=self.changeDiscreteFunction)

        self.getInfos()
        if currentTab == self:
            self.showSelf(upInfos=True, upImage=True)
    
    def changeDiscreteFunction(self, element):
        self.discreteFunction=DiscreteFunction(element)

        self.getInfos()
        
        imageProcessingPanel.setButtons()

class TabFromDiscreteFunction(Tab):
    def __init__(self, name:str, parent:ctk.CTkFrame, discreteFunction:DiscreteFunction):
        super().__init__(name, parent)

        self.discreteFunction=discreteFunction
        self.image=pool.apply_async(getImageFromDiscreteFunction, (self.discreteFunction,), callback=self.changeImage)
        self.getInfos()
    
    def changeImage(self, element):
        self.image=element

        self.getInfos()
        if currentTab==self:
            self.showSelf(upImage=True)

class TabFromFunction(Tab):
    def __init__(self, name:str, parent:ctk.CTkFrame, discreteFunction:DiscreteFunction, function, args):
        super().__init__(name, parent)

        pool.apply_async(TabFromFunction.applyFunction, (discreteFunction, function, args), callback=self.changeDiscreteFunction)
    
    def changeDiscreteFunction(self, element):
        tab=TabFromDiscreteFunction(self.name, self.parent, element)
        tab.showSelf(upImage=True, upInfos=True, clicked=True)

        self.tabFrame.pack_forget()
        del self
    
    def applyFunction(discreteFunction, function, args):
        args["function"]=discreteFunction

        function(**args)

        return discreteFunction


class ImageInfosPanel:
    def __init__(self):
        self.frame=ctk.CTkFrame(rightContainer)
        self.frame.pack(side="top", fill="x")
        
        ctk.CTkLabel(self.frame, text="Image Characteristics :", font=titleFont).pack()
        
        self.infos={"Width":0, "Height":0, "Number of pixels":0, "Local Variance":0, "Gradient Energy":0, "High Frequency Ratio":2}

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

        self.functions={"Bruitage poivre et sel": (saltAndPaperNoising, "probability"),
                        "Bruitage aléatoire": (randomNoising, "minAdd", "maxAdd")}
    
    def destroyButtons(self):
        for widget in self.frame.winfo_children():
            widget.destroy()
    
    def setButtons(self):
        self.buttons={}

        ctk.CTkLabel(self.frame, text="Image Processing Actions :", font=titleFont).pack(padx=10)

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

        args=[None]+[float(k.get()) for k in self.buttons[key][2]]
        newArgs=dict(zip(inspect.signature(self.functions[key][0]).parameters, args))

        name=key+"; "+"; ".join([self.functions[key][k]+":"+str(args[k]) for k in range(1, len(self.functions[key]))])

        tab=TabFromFunction(name, currentTab.tabFrame, currentTab.discreteFunction, self.functions[key][0], newArgs)
        tab.showSelf()


def importNewImage():
    filepath=tk.filedialog.askopenfilename()
    if filepath=="": return
    filename=filepath.split("/")[-1]

    try:
        image=Image.open(filepath)
    except UnidentifiedImageError as e:
        #raise e
        addErrorMessage(f"File format .{filename.split(".")[1]} not supported")
        return

    tab=TabFromImage(filename.split(".")[0], tabsFrame, image)
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






window=ctk.CTk()
window.title="Projet Maths-Info S4"
window.geometry("1200x800")


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
imageContainer.pack(fill="both", expand=True)

imageEditingActionsFrame=ctk.CTkFrame(middleContainer)
imageEditingActionsFrame.pack(side="bottom", fill="x")


rightContainer=ctk.CTkFrame(panedWindow)
panedWindow.add(rightContainer, minsize=300, stretch="never")

imageInfosPanel=ImageInfosPanel()

imageProcessingPanel=ImageProcessingPanel()

if __name__=="__main__":
    pool=Pool()

    window.mainloop()