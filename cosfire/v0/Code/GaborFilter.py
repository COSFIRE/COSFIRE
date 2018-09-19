#import numpy as np
import cv2

# cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, ktype)
# ksize - size of gabor filter (n, n)
# sigma - standard deviation of the gaussian function
# theta - orientation of the normal to the parallel stripes
# lambda - wavelength of the sunusoidal factor
# gamma - spatial aspect ratio
# psi - phase offset
# ktype - type and range of values that each pixel in the gabor kernel can hold

def getGaborResponse(imagen,parametros,theta,lambd):
    Resp={}
    for i in range(theta.size):
        for j in range(lambd.size):
            g_kernel = cv2.getGaborKernel(parametros[0], parametros[1], theta[i], lambd[j], parametros[4], parametros[5], ktype=cv2.CV_32F)
            #cv2.imshow("dsadad",imagen)
            filtered_img = cv2.filter2D(imagen, cv2.CV_32F, g_kernel)
            Resp[(theta[i],lambd[j])]=filtered_img
    #Faltaria normalizar cada respuesta de Gabor
    return Resp
    