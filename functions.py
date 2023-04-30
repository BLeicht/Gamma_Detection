#!/usr/bin/python3

import sys
import cv2

import numpy as np
import matplotlib.pyplot as plt

from skimage import io, util

def main(imagePath="./testImages/peppers.png"):
    #SetSysSettings()
    image = GrayscaleImport(imagePath)
    #histArray, bins = CalcHist(image)
    #PlotHist(histArray, bins)

    #arrayPeaksGaps = DetectPeaksGaps(histArray)
    
    #isCorrectedImage, correctionDirection  = DetectIfGamma(arrayPeaksGaps)


    gammadImage = AdjustGamma(image, )
    histArray, bins = CalcHist(gammadImage)
    arrayPeaksGaps = DetectPeaksGaps(histArray)
    isCorrectedImage, correctionDirection = DetectIfGamma(arrayPeaksGaps)
    print(isCorrectedImage)
    print(correctionDirection)


# Adjusts the gamma of a given image array and returns
# the new image array
def AdjustGamma(imageArray, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(imageArray, table)


# Takes a peaks and gaps array from (DetectPeaksGaps())
# and outputs whether or not it's a gamma corrected image
# and the direction it's corrected (as a 1 or -1 for
# up and down gamma) as a tuple
def DetectIfGamma(arrayPeaksGaps):
    firstSectionIs = 0

    if (arrayPeaksGaps.min() == arrayPeaksGaps.max()):
        return (False, -1*firstSectionIs)

    inSecondSection = False

    i = 0
    while (0 <= i <= 255):
        if (not firstSectionIs):
            firstSectionIs = arrayPeaksGaps[i]
        else:
            if (inSecondSection):
                if (firstSectionIs == arrayPeaksGaps[i]):
                    return False
            else:
                if ((arrayPeaksGaps[i] != 0) and (firstSectionIs != arrayPeaksGaps[i])):
                    inSecondSection = True
        i += 1


    return (True, -1*firstSectionIs)


# Finds all peaks and gaps from the given histogram
# of pixel values by comparing each pixel value to its
# neighbors. If a value is found to be a peak it's
# assigned a 1 in the outpu --, and if it's a gap it's
# assigned a -1 in the output. 
def DetectPeaksGaps(histArray):
    arrayPeaksGaps = np.zeros(256, dtype='b')
   
    for i in range(1, len(histArray)-1):
        gap = ((histArray[i] == 0) and ((histArray[i+1] != 0) or (histArray[i-1] != 0)))
        if (gap):
            arrayPeaksGaps[i] = -1
            continue
   
        # The 1.25x is arbitrary -- it's just meant to rule out regular peaks in an image. 
        peak = ((histArray[i] > histArray[i-1]*1.25) and (histArray[i] > histArray[i+1]*1.25) and (histArray[i-1] > 0) and (histArray[i+1] > 0))
        if (peak):
            arrayPeaksGaps[i] = 1
            continue 

    return arrayPeaksGaps

# Takes a grayscale image input and returns the
# occurence of pixel values from [0..255] with
# the returned array's index being the value
# and the element at the index the # of occurences
def CalcHist(grayscaleImageArray):
    maxVal = grayscaleImageArray.max()
    minVal = grayscaleImageArray.min()
    bins = maxVal - minVal + 1
    imageHist, bins = np.histogram(grayscaleImageArray, bins)
    return imageHist, bins

# Plots the incoming histogram (stored as array)
# with the given number of bins
def PlotHist(imageHist, bins):
    plt.stairs(imageHist, bins, fill=True)
    plt.show()

# Imports an image as a grayscale, converting the values
# to be between [0..255]
def GrayscaleImport(imagePath):
    image = io.imread(imagePath, as_gray=True)
    return util.img_as_ubyte(image)

# Sets miscellaneous helpful settings
def SetSysSettings():
    np.set_printoptions(threshold=sys.maxsize)
