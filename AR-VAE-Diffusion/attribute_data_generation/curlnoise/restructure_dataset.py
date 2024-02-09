from PIL import Image
import os, sys
import numpy as np
import cv2
import pandas as pd
import glob
import tqdm

def get_pd(image):
    image = np.array(image)
    return image.sum() / image.size

def get_size(image):
    img = np.array(image, np.uint8)
    _,thresh = cv2.threshold(img,240,255,cv2.THRESH_BINARY)
    thresh = cv2.bitwise_not(thresh)
    

    element = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(5, 5))

    morph_img = thresh.copy()
    cv2.dilate(src=thresh, kernel=element, dst=morph_img, iterations=2)
    cv2.erode(src=morph_img, kernel=element, dst=morph_img, iterations = 1)

    contours,_ = cv2.findContours(morph_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    areas = [cv2.contourArea(c) for c in contours]
    sorted_areas = np.sort(areas)

    #bounding box (red)
    try:    
        cnt=contours[areas.index(sorted_areas[-1])] #the biggest contour
        (x,y),radius = cv2.minEnclosingCircle(cnt)
        center = (int(x),int(y))
        radius = int(radius)
        cv2.circle(img,center,radius,(0,255,0),2)

    except:
        radius = 0
    return radius**2

def resize(path, to):
    for item in tqdm.tqdm(sorted(glob.iglob(path))):
        im = Image.open(item)
        imResize = im.resize((128,128), Image.Resampling.LANCZOS)
        imResize.save(to + item.split("/")[-1], 'png')


def compute_attribute(attribute_function, path):
    vals = []
    for item in tqdm.tqdm(sorted(glob.iglob(path))):
        im = Image.open(item)
        vals.append(attribute_function(im))
    return vals

path = "../data/retained/imagefolder128/train3/*.png"
path_to = "../data/retained/imagefolder128/train/*.png"
attributes_dict = {"pixel_density": get_pd, "size": get_size}
data = {}
data["name"] = [file for file in sorted(glob.iglob(path_to))]
#resize(path, "../data/retained/imagefolder128/train/")

for attribute, attribute_function in attributes_dict.items():
    data[attribute] = compute_attribute(attribute_function, path_to)
dataframe = pd.DataFrame(data)
for attribute in attributes_dict.keys():
    dataframe[attribute] = (dataframe[attribute]- dataframe[attribute].mean()) / dataframe[attribute].std()
print(dataframe.head(10))
dataframe.to_csv("../data/retained/imagefolder128/train/data.csv")