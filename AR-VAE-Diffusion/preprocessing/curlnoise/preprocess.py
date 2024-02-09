"""
Preprocess the images in the training dataset.
  3 folders are maintained to manage data:
    all <- contains all data points
    retained/train <- all data points that pass the "filter". Used for training and validating model.
    discarded <- all data points that fail the filter.
Ensure that retained/train is empty when preprocessing to prevent issues of having the wrong dataset
stores in retained/train.
"""

import numpy as np
import glob
import cv2 as cv
import matplotlib.pyplot as plt
import tqdm
data_dir_name = "../data/retained/curl_noise_03/*.png"

data_store = "../data/retained/imagefolder128/train3/"
discarded_store = "../data/retained/imagefolder128/discarded/"
images = []
def analyze_images(images, plot=True):
    splits = np.array_split(images, 20)
    k = 1
    print(len(images))
    for split in splits:
        print(len(split))
        if plot:
            plt.figure(figsize=(20, 8.5))
            for i in range(50):
                plt.subplot(10, 5, i + 1)
                plt.axis("off")
                plt.imshow(split[np.random.choice(split.shape[0])][0], cmap="gray")
            plt.show()
        k += 1
        print(split[-1][2])
    splits4 = splits[3]
    splits4 = np.array_split(splits4, 10)
    for split in splits4:
        print(len(split))
        if plot:
            plt.figure(figsize=(20, 8.5))
            for i in range(50):
                plt.subplot(10, 5, i + 1)
                plt.axis("off")
                plt.imshow(split[np.random.choice(split.shape[0])][0], cmap="gray")
            plt.show()
    plt.imshow(splits4[4][0][0])
    plt.show()
    return splits4[3][0][2]


# Apply a threshold to every image. As long as a pixel is "somewhat gray" we can determine the cutoff for the
# intensity of "grayness" by lowering the 2nd parameter of cv.threshold. A lower value means that a pixel is considered
# active at a higher threshold
# Note that 0 = black, 255 = white
for file in tqdm.tqdm(glob.iglob(data_dir_name)):
    image = cv.imread(file, cv.IMREAD_GRAYSCALE)
    ret, thresed_image = cv.threshold(image, 190, 255, cv.THRESH_BINARY)
    image = (thresed_image, file, thresed_image.sum())
    images.append(image)
    
   

    import shutil
    if image[2] < 66546720:
        shutil.copy(image[1], data_store)
    else:
        shutil.copy(image[1], discarded_store)

images = np.array(images)
images = sorted(images, reverse=True, key=lambda x: x[2])
# Used to get the decision boundary for rejecting images 66769200. Analyzed visually
# print(analyze_images(images, plot=True))