import customtkinter as ctk
import tkinter as tk
from PIL import Image

from Mathematics import *
from Methods import *
from Noising import *

nextAvailableId=1
currentId=0

lastSize=(0,0)

class Tab:
    def __init__(self, filepath:str, id):
        self.id=id
        self.name=filepath.split("/")[-1].split(".")[0]

        try:
            self.image=Image.open(filepath)
        except Exception as e:
            print(e)
            print("Error: incorrect file format")
        self.discreteFunction=DiscretefunctionFromImage(filepath)

        self.button=ctk.CTkButton(tabsFrame, text=self.name, command=lambda: showCurrentImage(self.id))

    def pack(self):
        self.button.pack(side="top", fill="both", expand=False)

def importImage():
    global nextAvailableId

    filepath=tk.filedialog.askopenfilename()
    if filepath=="":
        return
    
    tab=Tab(filepath, nextAvailableId)
    nextAvailableId+=1
    tabs.append(tab)

    importImageButton.pack_forget()
    tab.pack()
    importImageButton.pack(side="top", fill="both", expand=False)


def showCurrentImage(id):
    global imageMenuImage, imageMenuContainer, currentId

    if id==0:
        imageMenuContainer.grid_forget()
        return

    for tab in tabs:
        if tab.id==id:
            break

    scalingFactor=min(imageMenuContainer.winfo_width()/tab.image.width, imageMenuContainer.winfo_height()/tab.image.height)

    imageMenuContainer.grid_forget()
    imageMenuImage=ctk.CTkImage(tab.image, tab.image, (tab.image.width*scalingFactor, tab.image.height*scalingFactor))
    imageMenuContainer=ctk.CTkLabel(imageMenuFrame, image=imageMenuImage, text="")
    imageMenuContainer.grid(row=0, column=0, sticky="nsew")

    currentId=id
    



window=ctk.CTk()
window.title = "Image processing project"
window.geometry("800x600")


contentPanedWindow=tk.PanedWindow(window, orient="horizontal")
contentPanedWindow.pack(fill="both", expand=True)


tabsFrame=ctk.CTkFrame(contentPanedWindow)
contentPanedWindow.add(tabsFrame, minsize=200)

importImageButton=ctk.CTkButton(tabsFrame, text="Import Image", command=importImage)
importImageButton.pack(side="top", fill="both", expand=False)


imageMenuFrame=ctk.CTkFrame(contentPanedWindow)
contentPanedWindow.add(imageMenuFrame)
imageMenuFrame.columnconfigure(0, weight=3)
imageMenuFrame.columnconfigure(1, weight=1)
imageMenuFrame.rowconfigure(0, weight=1)

imageMenuControlsContainer=ctk.CTkFrame(imageMenuFrame)
imageMenuControlsContainer.grid(row=0, column=1, sticky="ns")

imageMenuImage=None
imageMenuContainer=ctk.CTkLabel(imageMenuFrame, image=imageMenuImage, text="")
imageMenuContainer.grid(row=0, column=0, sticky="nsew")


tabs=[]

window.mainloop()