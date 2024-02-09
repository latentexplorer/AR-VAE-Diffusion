from PIL import Image
import os, sys
import numpy as np
import cv2
import pandas as pd
import glob
import tqdm
from io import BytesIO
import random
from copy import deepcopy

def get_structural_complexity(img_path, r = 3, num = 0):
    # load image as grayscale img and normalise values to range 0-1
    img = cv2.imread(img_path, 0)
    if img.shape != (128, 128):
        img = cv2.resize(img, (128,128), interpolation = cv2.INTER_AREA)

    #standardize image
    mean = np.mean(img)
    std = np.std(img)
    img = (img - mean)/std
    rows, cols = img.shape
    # uncompressed size
    img_size_uncompressed = img.size
    # initialise new np array
    # structural complexity is calculated on this new image
    new_img = np.full((rows, cols), -1.0, dtype=np.float32)
    for i in range(img.shape[0]//(r) + 1):
        y = i * (r)
        y_end = min(y + (r), img.shape[0] + 1)
        for j in range(img.shape[1]//(r) + 1):
            x = j * (r)
            x_end = min(x + (r), img.shape[1] + 1)
            # print(img[y:y_end,x:x_end])
            m = img[y:y_end, x:x_end].mean() 
            v = 0
            if m >= 1: v = 1.
            elif m >= 0 and m < 1: v=0.7
            elif m >= -1 and m < 0: v=0.35
            else: v=0
            new_img[y:y_end, x:x_end] = v
    new_img_png = Image.fromarray(np.uint8((new_img)* 255), mode='L')
    img_file = BytesIO()
    new_img_png.save(img_file, 'png')
    
    # while metadata is still is file size, it is usually constant for all images
    # thus as it doesn't affect the relative order of the attributes
    # it proabaly has little effect on the effectiveness of the attribute value
    # in our model
    img_size_compressed = img_file.tell() # - metadata_constant 
    
    return img_size_compressed/img_size_uncompressed

def get_image_color_num(image_path, tolerance=35):
    # Read the image
    img = cv2.imread(image_path)

    # Reshape the image to a 2D array of pixels
    pixels = img.reshape((-1, 3))
    colors = []
    color_count = 0

    # YUV luminance constants used to multiply tolerance,
    # Thus greater tolerance for colors with higher luminance
    red_tolerance = np.maximum(0.299 * tolerance, 1)
    green_tolerance = np.maximum(0.587 * tolerance, 1)
    blue_tolerance = np.maximum(0.114 * tolerance, 1)

    # Apply rounding to the pixels
    rounded_pixels = np.round(pixels / [blue_tolerance, green_tolerance, red_tolerance]) * [blue_tolerance, green_tolerance, red_tolerance]
    rounded_pixels = rounded_pixels.astype("int32")
    unique_indices = np.unique(rounded_pixels.view([('', rounded_pixels.dtype)] * rounded_pixels.shape[1]))
    
    # Count the number of color groups
    color_count = len(unique_indices)
    return color_count

def compute_attribute(attribute_function, path):
    vals = []
    for item in tqdm.tqdm(sorted(glob.iglob(path))):
        vals.append(attribute_function(item))
    return vals

if __name__ == "__main__":
    path = "/home/anonymized/ASLR_DiffuseVAE/results/abstractart_ddpm/completed_training_color_70/200/images/*.png"
    attributes_dict = {"structural_complexity": get_structural_complexity, "color_diversity": get_image_color_num}
    data = {}
    data["name"] = [file for file in sorted(glob.iglob(path))]
    for attribute, attribute_function in attributes_dict.items():
        print(f"Computing {attribute}")
        data[attribute] = compute_attribute(attribute_function, path)
    dataframe = pd.DataFrame(data)
    for attribute in attributes_dict.keys():
        dataframe[attribute] = (dataframe[attribute]- dataframe[attribute].mean()) / dataframe[attribute].std()
    print(dataframe.head(10))
    dataframe.to_csv("/home/anonymized/ASLR_DiffuseVAE/results/csvs/data_70.csv")


