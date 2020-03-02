import pathlib
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os


# Glob the training data and load a single image path
current_dir = pathlib.Path.cwd()
home_dir = pathlib.Path.home()

print(current_dir)
print(home_dir)

training_paths = current_dir.glob('*.png')
training_sorted = [x for x in training_paths]



for i in training_sorted:


    im_path = i
    print(im_path)
    im = imageio.imread(str(im_path))
    # Print the image dimensions
    print('Original image shape: {}'.format(im.shape))




    def rgb2gray(rgb):
        return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
    im_gray = rgb2gray(im)

    print('New image shape: {}'.format(im_gray.shape))

    from skimage.filters import threshold_otsu
    thresh_val = threshold_otsu(im_gray)
    mask = np.where(im_gray > thresh_val, 1, 0)
    cv2.imwrite('%s.jpeg' % (im_path),mask*255)
# Make sure the larger portion of the mask is considered background
    if np.sum(mask==0) < np.sum(mask==1):
        mask = np.where(mask, 0, 1)
        print()
