'''
Created on 7 may. 2020

@author: David Martínez
'''

import numpy as np
import cv2 as cv
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
import pandas as pd
from openpyxl import load_workbook # Agregar nuevos datos a un archivo de excel.
from os import scandir, getcwd # Lectura de directorios
from matplotlib.pyplot import figure
from scipy.spatial import distance 

# Función para leer los archivos de un directorio.
def ImagesList(direccion = getcwd()):
    return [arch.name for arch in scandir(direccion) if arch.is_file()]

# Dirección de la carpeta para enlistar archivos.
dir_images = 'Database'

# Obtención de la lista de archivos.
listfiles = ImagesList(dir_images)

# -> Lectura de la imagen principal a comparar.
img = cv.imread('Database/Fruit23.jpg')
hist = cv.calcHist( [img], [0, 1, 2], None, [8 ,8, 8], [0, 256, 0, 256, 0, 256] )
hist = cv.normalize(hist, hist, 1.0, 0, cv.NORM_L1).flatten()

data = pd.DataFrame()
data['Path'] = listfiles 

datalist = [ ] # Inicializar una lista.
ManhattanDist = [ ]
EuclideanDist = [ ]

for i in range(0 , len(listfiles), 1):
    img_aux = cv.imread( dir_images + '\\' + str( listfiles[i] ))
    hist_aux = cv.calcHist( [img_aux], [0, 1, 2], None, [8 ,8, 8], [0, 256, 0, 256, 0, 256] )
    hist_aux = cv.normalize(hist_aux, hist_aux, 1.0, 0, cv.NORM_L1).flatten()
    datalist.append( cv.compareHist(hist, hist_aux, cv.HISTCMP_HELLINGER) ) 
    ManhattanDist.append( distance.cityblock( hist, hist_aux) )
    EuclideanDist.append( distance.euclidean(hist, hist_aux) )
    
    
data['Bhattacharyya'] = datalist
data['Manhattan distance'] = ManhattanDist
data['Euclidean distance'] = EuclideanDist

#data = data.sort_values('Bhattacharyya')
data = data.sort_values('Manhattan distance')
print(data)

# Pasar la dirección de las imágenes ordenadas a una lista.
listfiles = data['Path'].values.tolist()

color = ('r','g','b')
hspace = 0.6

figure('Original Image', figsize=(6,3))
plt.subplot(121)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(img)
plt.title("Original Image")
plt.subplot(122)
for i,col in enumerate(color):
    histr = cv.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.title("Histogram")
plt.subplots_adjust( left = 0.1, bottom= 0.1, right= 0.98, top= 0.92, wspace= 0.2, hspace= hspace )
plt.margins(0.2)


data = data.sort_values('Manhattan distance')
listfiles = data['Path'].values.tolist()

plimg = []

for j in range(0,20,1):
    aux = ( cv.imread( str(dir_images)+'//'+str(listfiles[j]) ) )
    plb, plg, plr = cv.split(aux) 
    plimg.append( cv.merge([plr,plg,plb]) ) 

fig=plt.figure(figsize=(10, 8))
for n in range(1, 21,1):
    fig.add_subplot(4, 5, n)
    plt.imshow( plimg[n-1] )



sel = 0
tit = 1
'''
for x in range( 0,5,1):
    figure('Results Bhattacharyya - ' + str(x+1), figsize = (6,8))
    for i in range(0, 8, 1):
        if (i+1)% 2 == 0:
            plt.subplot(  int(str (52)+ str(i+1)) )
            for j,col in enumerate(color):
                histr = cv.calcHist([img_aux],[j],None,[256],[0,256])
                plt.plot(histr,color = col)
                plt.xlim([0,256])
            plt.title( "Histogram " + str(tit) )
            tit = tit + 1        
        else:
            plt.subplot( int(str (52)+ str(i+1)) )
            img_aux = cv.imread( dir_images + '\\' + str( listfiles[sel] ))
            img_aux = cv.cvtColor(img_aux, cv.COLOR_BGR2RGB)
            plt.imshow(img_aux)
            plt.title( str( listfiles[sel] ) )
            sel = sel + 1
    plt.subplots_adjust( left = 0.02, bottom= 0, right= 0.98, top= 0.94, wspace= 0.1, hspace= hspace )
    plt.margins(0.2)



data = data.sort_values('Manhattan distance')
print(data)

# Pasar la dirección de las imágenes ordenadas a una lista.
listfiles = data['Path'].values.tolist()


sel = 0
tit = 1

for x in range(0, 5, 1):
    figure('Results Manhattan - '+ str(x+1), figsize = (6,8))
    for i in range(0, 8, 1):
        if (i+1)% 2 == 0:
            plt.subplot(  int(str (52)+ str(i+1)) )
            for j,col in enumerate(color):
                histr = cv.calcHist([img_aux],[j],None,[256],[0,256])
                plt.plot(histr,color = col)
                plt.xlim([0,256])
            plt.title( "Histogram " + str(tit) )
            tit = tit + 1        
        else:
            plt.subplot( int(str (52)+ str(i+1)) )
            img_aux = cv.imread( dir_images + '\\' + str( listfiles[sel] ))
            img_aux = cv.cvtColor(img_aux, cv.COLOR_BGR2RGB)
            plt.imshow(img_aux)
            plt.title( str( listfiles[sel] ) )
            sel = sel + 1
    plt.subplots_adjust( left = 0.02, bottom= 0, right= 0.98, top= 0.94, wspace= 0.1, hspace= hspace )
    plt.margins(0.2)


data = data.sort_values('Euclidean distance')
print(data)

# Pasar la dirección de las imágenes ordenadas a una lista.
listfiles = data['Path'].values.tolist()

sel = 0
tit = 1

for x in range(0, 5, 1):
    figure('Results Euclidean - '+ str(x+1) , figsize = (6,8))
    for i in range(0, 8, 1):
        if (i+1)% 2 == 0:
            plt.subplot(  int(str (52)+ str(i+1)) )
            for j,col in enumerate(color):
                histr = cv.calcHist([img_aux],[j],None,[256],[0,256])
                plt.plot(histr,color = col)
                plt.xlim([0,256])
            plt.title( "Histogram " + str(tit) )
            tit = tit + 1        
        else:
            plt.subplot( int(str (52)+ str(i+1)) )
            img_aux = cv.imread( dir_images + '\\' + str( listfiles[sel] ))
            img_aux = cv.cvtColor(img_aux, cv.COLOR_BGR2RGB)
            plt.imshow(img_aux)
            plt.title( str( listfiles[sel] ) )
            sel = sel + 1
    plt.subplots_adjust( left = 0.02, bottom= 0, right= 0.98, top= 0.94, wspace= 0.1, hspace= hspace )
    plt.margins(0.2)
'''

plt.show()
