'''
Created on 28 abr. 2020

@author: David Martínez
'''
import numpy as np
import cv2 as cv
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
import pandas as pd
from openpyxl import load_workbook # Agregar nuevos datos a un archivo de excel.
from os import scandir, getcwd # Lectura de directorios
from cmath import sqrt
from builtins import int

# Función para obtener los valores RGB promedio de una imagen.
def Average_color( image):
    """
    This function obtain the average color
    of an image passed in as parameter.
    """
    y,x,z = image.shape # Obtener el número de filas, columnas y profundidad.
    xy = x*y # Total de pixeles por capa.
    
    b,g,r = cv.split(image) # Separar la imagen en capas.

    # Convertir las capas a vectores unidimensionales.
    b = b.flatten() 
    g = g.flatten()
    r = r.flatten()
    
    mean_r = 0
    mean_g = 0
    mean_b = 0

    for i in range(0,xy,1): # Conteo de los pixeles para el promedio.
        mean_r = r[i] + mean_r
        mean_g = g[i] + mean_g
        mean_b = b[i] + mean_b
        
    # Obtener el valor promedio de cada color.    
    mean_r = int(mean_r/xy)
    mean_g = int(mean_g/xy)
    mean_b = int(mean_b/xy)
    
    #Regresar los valores obtenidos.
    return mean_r, mean_g, mean_b  
    
    """
    print("Red ->",mean_r)
    print("Green ->",mean_g)
    print("Blue ->",mean_b)
    """      
    
# Nombre del archivo de excel.

name_file = 'AverageRGB-Database' 

img = cv.imread('Database/Fruit23.jpg')

r,g,b = Average_color( img )
print("Mean RGB of Image ->", r,g,b)

# Leer archivo .xlsx con los valores promedio rgb de cada imagen    
reader = pd.read_excel(name_file + '.xlsx')
#print(reader)

m_RED = reader['RED']
m_GREEN = reader['GREEN']
m_BLUE = reader['BLUE']

dist = sqrt( pow( r - int(m_RED[0]), 2)  + pow( g - int(m_GREEN[0]), 2) + pow( b - int(m_BLUE[0]), 2) ) 
distlist =[ float(dist.real) ] 

for i in range(1, len(reader), 1):
    dist = sqrt( pow( r - int(m_RED[i]), 2)  + pow( g - int(m_GREEN[i]), 2) + pow( b - int(m_BLUE[i]), 2) ) 
    distlist.append( float(dist.real) )

reader['Distance'] = distlist

reader = reader.sort_values('Distance')
print(reader)

writer = pd.ExcelWriter('Results.xlsx', engine='xlsxwriter')

# Convert the dataframe to an XlsxWriter Excel object.
reader.to_excel(writer, sheet_name='Sheet1', index=False)

# Close the Pandas Excel writer and output the Excel file.
writer.save()

"""
# Graficar los puntos de todas las imágenes.
rojo = m_RED.to_numpy()
verde = m_GREEN.to_numpy()
azul = m_BLUE.to_numpy()

plt.figure(1)
ax = plt.axes(projection='3d')
ax.scatter3D(azul, verde, rojo ,s=1.2 ,alpha=0.75);
ax.view_init(17,200)
ax.set_xlabel('Blue')
ax.set_ylabel('Green')
ax.set_zlabel('Red');
plt.show()
"""

listpaths = reader['Path'].values.tolist()

plimg = []

#cv.imshow("Original File", img)  
 
for j in range(0,20,1):
    aux = ( cv.imread( str(listpaths[j]) ) )
    plb, plg, plr = cv.split(aux) 
    plimg.append( cv.merge([plr,plg,plb]) ) 

fig=plt.figure(figsize=(10, 8))
for n in range(1, 21,1):
    fig.add_subplot(4, 5, n)
    plt.imshow( plimg[n-1] )
plt.show()  
 
cv.waitKey()


