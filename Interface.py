import customtkinter as ctk
import tkinter as tk
from PIL import Image, UnidentifiedImageError
import inspect
from functools import partial

from Mathematics import *
from Methods import *
from Noising import *

ERROR_SHOW_TIME=5000

nextAvailableId=1
tabs=[]

currentTab=None

class Tab:
    def __init__(self, name:str, image:Image.Image, parent:ctk.CTkFrame):
        global nextAvailableId

        self.id=nextAvailableId
        nextAvailableId+=1

        self.name=name

        self.image=image
        self.parent=parent

        self.childrenTabs=[]

        self.getDiscreteFunction()
        self.getInfos()
        self.createTabElement()

        tabs.append(self)
    
    def createTabElement(self):
        self.tabFrame=ctk.CTkFrame(self.parent)
        self.tabFrame.pack(fill="x", padx=(20*(self.parent!=tabsFrame),0))

        self.tabButton=ctk.CTkButton(self.tabFrame, text=self.name, command=lambda: self.showSelf())
        self.tabButton.pack(fill="x")
    
    def showSelf(self):
        global currentTab

        self.updateImage()

        imageInfosPanel.updateInfos(self.infos)

        currentTab=self
    
    def updateImage(self):
        global imageContainer

        scalingFactor=min(imageContainer.winfo_width()/self.image.width, imageContainer.winfo_height()/self.image.height)

        imageContainer.pack_forget()

        image=ctk.CTkImage(self.image, self.image, (self.image.width*scalingFactor, self.image.height*scalingFactor))
        imageContainer=ctk.CTkLabel(middleContainer, image=image, text="")

        imageContainer.pack(fill="both", expand=True)
    
    def getDiscreteFunction(self):
        coeffs=(0.299, 0.587, 0.114)
        imageMat=[]

        for j in range(self.image.height):
            imageMat.append([])
            for i in range(self.image.width):
                pixelColors=self.image.getpixel((i,j))
                grayLevel=round(sum([coeffs[k]*pixelColors[k] for k in range(3)]))
                imageMat[j].append(grayLevel)
                self.image.putpixel((i,j), (grayLevel, grayLevel, grayLevel))
        
        self.discreteFunction=DiscreteFunction(imageMat)
    
    def getInfos(self):
        self.infos={}

        self.infos["Width"]=self.discreteFunction.width
        self.infos["Height"]=self.discreteFunction.height
        self.infos["Number of pixels"]=self.discreteFunction.width*self.discreteFunction.height
    
    def addChildTab(self, name:str, image:Image.Image, showChild:bool):
        tab=Tab(name, image, self.tabFrame)
        self.childrenTabs.append(tab)

        if showChild: tab.showSelf()
    
    
class ImageInfosPanel:
    def __init__(self):
        self.frame=ctk.CTkFrame(rightContainer)
        self.frame.pack(side="top", fill="x")
        
        ctk.CTkLabel(self.frame, text="Image Characteristics :", font=titleFont).pack()
        
        self.infos=("Width", "Height", "Number of pixels")

        self.setInfos()
    
    def setInfos(self):
        self.infosLabels={}

        for k in self.infos:
            label=ctk.CTkLabel(self.frame, text=k+": -")
            label.pack()
            self.infosLabels[k]=label

    def updateInfos(self, newInfos:dict):
        for key, value in self.infosLabels.items():
            if key in newInfos.keys():
                value.configure(text=key+": "+str(newInfos[key]))
            else:
                value.configure(text=key+": -")


class ImageProcessingPanel:
    def __init__(self):
        self.frame=ctk.CTkFrame(rightContainer)
        self.frame.pack(side="top", fill="x", pady=(5,0))

        ctk.CTkLabel(self.frame, text="Image Processing Actions :", font=titleFont).pack(padx=10)

        self.functions={"Bruitage poivre et sel": (saltAndPaperNoising, "probability"),
                        "Bruitage aléatoire": (randomNoising, "minAdd", "maxAdd")}

        self.setButtons()
    
    def setButtons(self):
        self.buttons={}

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
        newDiscreteFunction=currentTab.discreteFunction.copy()

        args=[newDiscreteFunction]+[float(k.get()) for k in self.buttons[key][2]]
        
        newArgs=dict(zip(inspect.signature(self.functions[key][0]).parameters, args))
        self.functions[key][0](**newArgs)

        name=key+"; "+"".join([self.functions[key][k]+":"+str(args[k])+"; " for k in range(1, len(self.functions[key]))])
        image=getImageFromDiscreteFunction(newDiscreteFunction)
        currentTab.addChildTab(name, image, True)


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

    tab=Tab(filename.split(".")[0], image, tabsFrame)
    tab.showSelf()



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
window.geometry("800x600")


titleFont=ctk.CTkFont("font1", 13, "bold", underline=True)


panedWindow=tk.PanedWindow(window, orient="horizontal", background="black")
panedWindow.pack(fill="both", expand=True)


errorMessageLabel=ctk.CTkLabel(window, text="")


leftContainer=ctk.CTkFrame(panedWindow)
panedWindow.add(leftContainer, minsize=200, stretch="never")

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
panedWindow.add(rightContainer, minsize=100, stretch="never")

imageInfosPanel=ImageInfosPanel()

imageProcessingPanel=ImageProcessingPanel()

window.mainloop()