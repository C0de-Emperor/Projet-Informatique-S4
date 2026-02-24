from PIL import Image

a=Image.open("InterfaceData/delete.png")
b=Image.new("RGBA", a.size)

for i in range(a.width):
    for j in range(a.height):
        value=a.getpixel((i,j))
        if sum([k<=50 for k in tuple(value)]) == 3:
            b.putpixel((i,j), (255,255,255,255))
        else:
            b.putpixel((i,j), (0,0,0,0))

b.save("InterfaceData/delete.png")