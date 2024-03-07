'''
Created on 13 jun. 2020

@author: David Martínez
'''

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance 
import pandas as pd
import time as t
from os import scandir, getcwd # Lectura de directorios
from numpy.core.numeric import infty, Infinity

# Función para leer los archivos de un directorio.
def ImagesList(direccion = getcwd()):
    return [arch.name for arch in scandir(direccion) if arch.is_file()]

# Dirección de la carpeta para enlistar archivos.
dir_images = 'Database'

# Obtención de la lista de archivos.
listfiles = ImagesList(dir_images)

img = cv.imread('Database/CASA11.jpg')
imggray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

sift = cv.xfeatures2d.SIFT_create() # Crear la clase SIFT para extraer descriptores.
bf = cv.BFMatcher()

kp , des = sift.detectAndCompute( img, None )

imgdist = []

for i in range(0 , len(listfiles) ): #len(listfiles)
    imgaux = cv.imread( str(dir_images) + '\\' + str( listfiles[i] )  )
    kpaux , desaux = sift.detectAndCompute( imgaux , None ) 
    matches = bf.knnMatch(des,desaux, k=2)
    
    good = [] # Lista de las mejores coincidencias entre dos imágenes.
    dist = 0
    
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])
            dist = m.distance + dist
    
    try:
        da = dist/len(good)
    except ZeroDivisionError:
        da = Infinity
    
    imgdist.append( len(good) )
    print(len(good))    


df = pd.DataFrame()

df["Files"] = listfiles
df["Dist"] = imgdist
print(listfiles)
df = df.sort_values("Dist",ascending=False)
listfiles = df["Files"].tolist()

print(listfiles)

plimg = []

#cv.imshow("Original File", img)  
 
for j in range(0,20,1):
    aux =  cv.imread( str(dir_images) +'\\' + str(listfiles[j]) ) 
    plb, plg, plr = cv.split(aux) 
    plimg.append( cv.merge([plr,plg,plb]) ) 

fig=plt.figure(figsize=(10, 8))
for n in range(1, 21,1):
    fig.add_subplot(4, 5, n)
    plt.imshow( plimg[n-1] )
plt.show()  
 
cv.waitKey()


