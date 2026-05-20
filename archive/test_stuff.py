
import cv2
import numpy as np

from src.utils import apply_colour_threshold, apply_closing

from src.config import load_config, set_global_config
config = load_config("config.json")
set_global_config(config)


img_path = "data/reference_images/cropped/wild.jpg"
img_path = "data/train_images/L1000912.jpg"
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img_black = apply_colour_threshold(img_rgb, color="k")

img_black_closed = apply_closing(img_black, disk_size=3.5)

# plot the original and the black mask
import matplotlib.pyplot as plt 

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off") 

plt.subplot(1, 2, 2)
plt.imshow(img_black_closed, cmap="gray")
plt.title("Black Mask")
plt.axis("off") 

plt.show()
