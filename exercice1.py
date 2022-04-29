from cProfile import label
from cmath import tan
import numpy as np
import cv2
from matplotlib import pyplot as plt

# recupertion de la map des markers
def getMarkers(im):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(41,25))
    erosion = cv2.erode(im,kernel,iterations=1)
    num_label, centers = cv2.connectedComponents(erosion)
    normalize_thresh = (thresh/np.amax(thresh)).astype(np.uint8) # normalisation de thresh pour avoir des valeurs de fond = 0 et taches = 1
    labels = centers + normalize_thresh # on associe les elements

    return labels

# recuperation de la map de priorites
def getPrio(im):
    dist = cv2.distanceTransform(im,cv2.DIST_L2,5)
    neg = (np.amax(dist)-dist)/np.amax(dist)*255
    neg = neg.astype(np.uint8)
    binary_im = (im/np.amax(im)).astype(np.uint8)
    dist = cv2.multiply(neg,binary_im)

    return dist

# initialisation de la FAH
def initFAH(prio,labels):
    lignes,colonnes = np.shape(prio)
    FAH_x = [[] for i in range(256)]
    FAH_y = [[] for i in range(256)]

    for i in range(lignes):
        for j in range(colonnes):
            if labels[i][j] != 1:
                ind = prio[i][j]
                FAH_x[ind].append(i)
                FAH_y[ind].append(j)

    return FAH_x,FAH_y

# récupération des voisins autour d'un pixel
def getVoisins(labels,i,j):
    lignes,colonnes = np.shape(labels)
    ret_x = []
    ret_y = []

    if i < lignes-1:
        if labels[i+1][j] == 1:
            ret_x.append(i+1)
            ret_y.append(j)
    if i > 0:
        if labels[i-1][j] == 1:
            ret_x.append(i-1)
            ret_y.append(j)
    if j < colonnes-1:
        if labels[i][j+1] == 1:
            ret_x.append(i)
            ret_y.append(j+1)
    if j > 0:
        if labels[i][j-1] == 1:
            ret_x.append(i)
            ret_y.append(j-1)

    return ret_x,ret_y

## Process
im = cv2.imread("smarties.png",0)

ret,thresh = cv2.threshold(im,250,255,cv2.THRESH_BINARY_INV)

labels = getMarkers(thresh)
prio = getPrio(thresh)

FAH_x,FAH_y = initFAH(prio,labels)
vide = [[] for i in range(256)]

i = 0
lim = len(FAH_x)
process_labels = labels

while FAH_x != vide:
    if FAH_x[i] != []:
        x = FAH_x[i][0]
        y = FAH_y[i][0]
        lab = process_labels[x][y]
        FAH_x[i].pop(0)
        FAH_y[i].pop(0)

        voisins_x,voisins_y = getVoisins(labels,x,y)

        if voisins_x != []:
            for k in range(len(voisins_x)):
                v_x = voisins_x[k]
                v_y = voisins_y[k]
                process_labels[v_x][v_y] = lab
                ind = prio[v_x][v_y]
                FAH_x[ind].append(v_x)
                FAH_y[ind].append(v_y)
    else:
        i = i+1
        if i == lim and FAH_x != vide:
            i = 0

## Affichage
plt.figure(1)
plt.subplot(221)
plt.imshow(im,'gray')
plt.title('Image originale')
plt.subplot(222)
plt.imshow(thresh,'gray')
plt.title('Image seuillé')
plt.subplot(223)
plt.imshow(labels)
plt.title('Image labélisé')
plt.subplot(224)
plt.imshow(prio,'gray')
plt.title('Priorités')

plt.figure(2)
plt.imshow(process_labels)
plt.title('Image segmentée')

plt.show()