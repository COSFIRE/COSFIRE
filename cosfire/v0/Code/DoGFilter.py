#import numpy as np
import cv2

def get_difference_of_gaussians_response(imagen, paremetros, sigma):
    response={}
    for i in range(len(sigma)):
        #cv2.imshow("dfsf",imagen)
        g1=cv2.GaussianBlur(imagen, (3,3), sigma[i])
        g2=cv2.GaussianBlur(imagen, (3,3), 0.5*sigma[i])
        r=g1-g2
        response[sigma[i]]=r
    return response