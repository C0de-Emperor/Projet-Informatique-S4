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
    def __init__(self, name:str, parent:ctk.CTkFrame):
        global nextAvailableId

        self.id=nextAvailableId
        nextAvailableId+=1

        self.image:Image.Image=None
        self.discreteFunction:DiscreteFunction=None

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

        fileEditingPanel.updateName(self.name)

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
            self.infos["Local Variance"]=mainPool.apply_async(getInfoForCallback, (self.discreteFunction, "Local Variance"), callback=self.changeInfo)
            self.infos["Gradient Energy"]=mainPool.apply_async(getInfoForCallback, (self.discreteFunction, "Gradient Energy"), callback=self.changeInfo)
            self.infos["High Frequency Ratio"]=mainPool.apply_async(getInfoForCallback, (self.discreteFunction, "High Frequency Ratio"), callback=self.changeInfo)
    
    def changeInfo(self, element):
        self.infos[element[1]]=element[0]

        if currentTab==self:
            self.showSelf(upInfos=True)

class TabFromImage(Tab):
    def __init__(self, name:str, parent:ctk.CTkFrame, image:Image.Image):
        super().__init__(name, parent)
        
        self.image=mainPool.apply_async(getGrayScaleImage, (image, (0.299, 0.587, 0.114)), callback=self.changeImage)
    
    def changeImage(self, element):
        self.image=element

        self.discreteFunction=mainPool.apply_async(getKernelFromImage, (self.image, (0.299, 0.587, 0.114)), callback=self.changeDiscreteFunction)

        self.getInfos()
        if currentTab == self:
            self.showSelf(upInfos=True, upImage=True)
    
    def changeDiscreteFunction(self, element):
        self.discreteFunction=DiscreteFunction(element)

        self.getInfos()
        
        self.parent.after(0, imageProcessingPanel.setButtons)

class TabFromDiscreteFunction(Tab):
    def __init__(self, name:str, parent:ctk.CTkFrame, discreteFunction:DiscreteFunction):
        super().__init__(name[1:], parent)

        self.discreteFunction=discreteFunction
        self.image=mainPool.apply_async(getImageFromDiscreteFunction, (self.discreteFunction,), callback=self.changeImage)
        self.getInfos()
    
    def changeImage(self, element):
        self.image=element

        self.getInfos()
        if currentTab==self:
            self.showSelf(upImage=True)

class TabFromFunction(Tab):
    def __init__(self, name:str, parent:ctk.CTkFrame, discreteFunction:DiscreteFunction, function, args):
        super().__init__("*"+name, parent)

        functionProcess=mainPool.apply_async(TabFromFunction.applyFunction, (discreteFunction, function, args), callback=self.changeDiscreteFunction)
    
    def changeDiscreteFunction(self, element):
        tab=TabFromDiscreteFunction(self.name, self.parent, element)
        tab.showSelf(upImage=True, upInfos=True, clicked=True)

        self.tabFrame.pack_forget()
        del self
    
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
                        "Bruitage aléatoire": (randomNoising, "minAdd", "maxAdd"),
                        "FFT2": (DiscreteFunctionFFT2Module,),
                        "Filtre médian": (DiscreteFunction.medianFilter, "rayon")}
    
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

        args=[float(k.get()) for k in self.buttons[key][2]]

        name=key+"; "+"; ".join([self.functions[key][k]+":"+str(args[k-1]) for k in range(1, len(self.functions[key]))])

        tab=TabFromFunction(name, currentTab.tabFrame, currentTab.discreteFunction, self.functions[key][0], args)
        tab.showSelf(upImage=True, upInfos=True)

class FileEditingPanel:
    def __init__(self):
        self.frame=ctk.CTkFrame(middleContainer, height=50)
        self.frame.pack(side="bottom", fill="x")
        self.frame.rowconfigure(0, weight=1)
        self.frame.columnconfigure(0, weight=2)

        self.fileNameTextBox=ctk.CTkEntry(self.frame)
        self.fileNameTextBox.grid(row=0, column=0, sticky="we")

        saveImageImage=ctk.CTkImage(Image.open("InterfaceData/rename.png"), Image.open("InterfaceData/rename.png"))
        self.renameButton=ctk.CTkButton(self.frame, image=saveImageImage, text="", width=saveImageImage._size[0], height=saveImageImage._size[1], command=self.renameImage)
        self.renameButton.grid(row=0, column=1, sticky="e")

        saveImageImage=ctk.CTkImage(Image.open("InterfaceData/save.png"), Image.open("InterfaceData/save.png"))
        self.saveButton=ctk.CTkButton(self.frame, image=saveImageImage, text="", width=saveImageImage._size[0], height=saveImageImage._size[1], command=self.saveImage)
        self.saveButton.grid(row=0, column=2, sticky="e")

        copyImageImage=ctk.CTkImage(Image.open("InterfaceData/copy.png"), Image.open("InterfaceData/copy.png"))
        self.copyImageButton=ctk.CTkButton(self.frame, image=copyImageImage, text="", width=copyImageImage._size[0], height=copyImageImage._size[1], command=self.copyImage)
        self.copyImageButton.grid(row=0, column=3, sticky="e")

        deleteImageImage=ctk.CTkImage(Image.open("InterfaceData/delete.png"), Image.open("InterfaceData/delete.png"))
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

    def copyImage(self):
        copy(currentTab.image)
    
    def deleteImage(self):
        global currentTab, imageContainer

        if currentTab==None: return

        for k in vars(currentTab):
            print(k, type(k))
            if type(k)==pool.AsyncResult:
                print("blub")
                k.terminate()
        
        for k in currentTab.infos.values():
            print(k, type(k))
            if type(k)==pool.AsyncResult:
                print("blub")
                k.terminate()
        
        currentTab.tabFrame.pack_forget()
        tabs.remove(currentTab)

        if len(tabs)!=0: 
            currentTab=tabs[0]
            currentTab.showSelf(upImage=True,  upInfos=True, clicked=True)
        else: 
            currentTab=None

            imageContainer.pack_forget()
            imageContainer=ctk.CTkLabel(middleContainer, text="Image visualisation window")
            imageContainer.pack(side="top", fill="both", expand=True)

            imageInfosPanel.updateInfos({})
            imageProcessingPanel.destroyButtons()
            self.updateName("")


def DiscreteFunctionFFT2Module(discreteFunction:DiscreteFunction) -> DiscreteFunction:
    #if discreteFunction.width*discreteFunction.height > 512**2: fft2Kernel=FFT2Boost(discreteFunction.kernel)
    #else: fft2Kernel=FFT2(discreteFunction.kernel)
    fft2Kernel=FFT2(discreteFunction.kernel)

    FFT2DiscreteFunction=ComplexDiscreteFunction(fft2Kernel)
    FFT2DiscreteFunctionModule=FFT2DiscreteFunction.getModule(True)
    FFT2DiscreteFunctionModule.resizeAmplitude()
    #FFT2DiscreteFunctionModuleRevolved=FFT2DiscreteFunctionModule.getRevolve()

    return FFT2DiscreteFunctionModule#Revolved

def importImageFromClipboard(event):
    image=paste()
    if image==None: return

    tab=TabFromImage("image", tabsFrame, image)
    tab.showSelf(upImage=True, upInfos=True)

def importNewImage():
    filepath=tk.filedialog.askopenfilename(title="Select File", filetypes=(("png files","*.png"),("jpeg files","*.jpg"),("all files","*.*")))
    if filepath=="": return
    filename=os.path.basename(filepath)

    try:
        image=Image.open(filepath)
    except UnidentifiedImageError as e:
        #raise e
        addErrorMessage(f"File format .{filename.split(".")[1]} not supported")
        return

    tab=TabFromImage(filename.split(".")[:-1], tabsFrame, image)
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
window.geometry("1200x800")
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


rightContainer=ctk.CTkFrame(panedWindow)
panedWindow.add(rightContainer, minsize=300, stretch="never")

imageInfosPanel=ImageInfosPanel()

imageProcessingPanel=ImageProcessingPanel()

if __name__=="__main__":
    mainPool=Pool()

    window.mainloop()