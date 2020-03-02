
import os
import pathlib
import imageio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

current_dir = pathlib.Path.cwd()

for i in range(1,21):
    image_path = os.path.join(current_dir, "%i", "images", "%i") + ".jpeg"

import os.path

base_path = pathlib.Path.cwd()
directory_generator = os.walk(base_path)
next(directory_generator)
path_tree = {}
for directories in directory_generator:
    path_tree[os.path.basename(root_path)] = [
        os.path.join(root_path, file_path) for file_path in files]
    print(path_tree)
