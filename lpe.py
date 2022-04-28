import numpy as np
import cv2
from matplotlib import pyplot as plt
import math





IM = cv2.imread('smarties.png',0)


ret,thresh1 = cv2.threshold(IM,250,255,cv2.THRESH_BINARY_INV)


S = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(37,18)) #élément sous forme d'ellipse
I_erod=cv2.erode(thresh1,S,iterations=1) #érosion
seuil=255-I_erod
affich=seuil+thresh1


ret, labels = cv2.connectedComponents(I_erod) #explication
plt.figure()
plt.imshow(labels+affich-254,'brg')
plt.show()