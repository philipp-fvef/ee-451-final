import numpy as np

from skimage.color import rgb2hsv
from skimage.morphology import closing, opening, disk, remove_small_holes, remove_small_objects, binary_dilation
import cv2

from sklearn.metrics.pairwise import euclidean_distances
from skimage.measure import regionprops

def extract_rgb_channels(img):
    """
    Extract RGB channels from the input image.

    Args
    ----
    img: np.ndarray (M, N, C)
        Input image of shape MxN and C channels.

    Return
    ------
    data_red: np.ndarray (M, N)
        Red channel of input image
    data_green: np.ndarray (M, N)
        Green channel of input image
    data_blue: np.ndarray (M, N)
        Blue channel of input image
    """

    # Get the shape of the input image
    M, N, C = np.shape(img)

    # print(f"Input image shape: {M}x{N} with {C} channels")

    # Define default values for RGB channels
    data_red = np.zeros((M, N))
    data_green = np.zeros((M, N))
    data_blue = np.zeros((M, N))

    # Extract RGB channels
    data_red = img[:, :, 0]
    data_green = img[:, :, 1]
    data_blue = img[:, :, 2]

    return data_red, data_green, data_blue


def extract_hsv_channels(img):
    """
    Extract HSV channels from the input image.

    Args
    ----
    img: np.ndarray (M, N, C)
        Input image of shape MxN and C channels.

    Return
    ------
    data_h: np.ndarray (M, N)
        Hue channel of input image
    data_s: np.ndarray (M, N)
        Saturation channel of input image
    data_v: np.ndarray (M, N)
        Value channel of input image
    """

    # Get the shape of the input image
    M, N, C = np.shape(img)

    # Define default values for HSV channels
    data_h = np.zeros((M, N))
    data_s = np.zeros((M, N))
    data_v = np.zeros((M, N))

    # use rgb2hsv function
    img_hsv = rgb2hsv(img)
    data_h = img_hsv[:, :, 0]
    data_s = img_hsv[:, :, 1]
    data_v = img_hsv[:, :, 2]

    return data_h, data_s, data_v


def apply_rgb_threshold(img, r_min=0, r_max=255, g_min=0, g_max=255, b_min=0, b_max=255):
    """
    Apply threshold to input image.

    Args
    ----
    img: np.ndarray (M, N, C)
        Input image of shape MxN and C channels.

    Return
    ------
    img_th: np.ndarray (M, N)
        Thresholded image.
    """

    # Define the default value for the input image
    M, N, C = np.shape(img)
    img_th = np.zeros((M, N))

    # Use the previous function to extract RGB channels
    data_red, data_green, data_blue = extract_rgb_channels(img=img)

    # Apply threshold to each channel
    img_th = (
        (data_red >= r_min) 
        & (data_red <= r_max) 
        & (data_green >= g_min) 
        & (data_green <= g_max) 
        & (data_blue >= b_min) 
        & (data_blue <= b_max)
    )

    return img_th


def apply_hsv_threshold(img, h_min=0.0, h_max=1.0, s_min=0.0, s_max=1.0, v_min=0.0, v_max=1.0):
    """
    Apply threshold to the input image in hsv colorspace.

    If min is bigger than max, the threshold will be applied in a circular way. 
    For example, if h_min=0.8 and h_max=0.2, 
    the threshold will be applied to the hue values that are 
    either greater than 0.8 or smaller than 0.2.

    Args
    ----
    img: np.ndarray (M, N, C)
        Input image of shape MxN and C channels.

    Return
    ------
    img_th: np.ndarray (M, N)
        Thresholded image.
    """

    # Define the default value for the input image
    M, N, C = np.shape(img)
    img_th = np.zeros((M, N))

    # Use the previous function to extract HSV channels
    data_h, data_s, data_v = extract_hsv_channels(img=img)

    # Apply threshold to each channel, taking into account the circular nature of hue values
    if h_min < h_max:
        img_th = (
            (data_h >= h_min) & (data_h <= h_max) & (data_s >= s_min) & (data_s <= s_max) & (data_v >= v_min) & (data_v <= v_max)
        )
    else:
        img_th = (
            ((data_h >= h_min) | (data_h <= h_max)) & (data_s >= s_min) & (data_s <= s_max) & (data_v >= v_min) & (data_v <= v_max)
        )
    return img_th


def apply_closing(img_th, disk_size):
    """
    Apply closing to input mask image using disk shape.
    Closing is a dilation followed by an erosion. 
    It can be used to close small holes in the image.

    Args
    ----
    img_th: np.ndarray (M, N)
        Image mask of size MxN.
    disk_size: int
        Size of the disk to use for opening

    Return
    ------
    img_closing: np.ndarray (M, N)
        Image after closing operation
    """

    # Define default value for output image
    img_closing = np.zeros_like(img_th)

    closing_disk = disk(disk_size)
    img_closing = closing(img_th, footprint=closing_disk)

    return img_closing


def apply_opening(img_th, disk_size):
    """
    Apply opening to input mask image using disk shape.
    Opening is an erosion followed by a dilation.
    It can be used to remove small objects from the image.

    Args
    ----
    img_th: np.ndarray (M, N)
        Image mask of size MxN.
    disk_size: int
        Size of the disk to use for opening

    Return
    ------
    img_opening: np.ndarray (M, N)
        Image after opening operation
    """

    # Define default value for output image
    img_opening = np.zeros_like(img_th)

    opening_disk = disk(disk_size)
    img_opening = opening(img_th, footprint=opening_disk)

    return img_opening

def remove_holes(img_th, size):
    """
    Remove holes from input image that are smaller than size argument.

    Args
    ----
    img_th: np.ndarray (M, N)
        Image mask of size MxN.
    size: int
        Minimal size of holes

    Return
    ------
    img_holes: np.ndarray (M, N)
        Image after remove holes operation
    """

    # Define default value for input image
    img_holes = np.zeros_like(img_th)

    img_holes = remove_small_holes(img_th, area_threshold=size)

    return img_holes


def remove_objects(img_th, size):
    """
    Remove objects from input image that are smaller than size argument.

    Args
    ----
    img_th: np.ndarray (M, N)
        Image mask of size MxN.
    size: int
        Minimal size of objects

    Return
    ------
    img_obj: np.ndarray (M, N)
        Image after remove small objects operation
    """

    # Define default value for input image
    img_obj = np.zeros_like(img_th)

    img_obj = remove_small_objects(img_th, min_size=size)

    return img_obj


def find_contours(images: np.ndarray, n: int = 0):
    """
    Find the contours for the set of images
    
    Args
    ----
    images: np.ndarray (N, H, W) or (H, W)
        Source images to process

    Return
    ------
    contours: list of np.ndarray
        List of N arrays containing the coordinates of the contour.
    """
    images = np.asarray(images)
    if images.ndim == 2:
        images = images[np.newaxis, ...]
    elif images.ndim != 3:
        raise ValueError("images must have shape (H,W) or (N,H,W)")

    N = images.shape[0]
    contours = []

    for i in range(N):
        img = images[i]
        # normalize to uint8 single-channel (handles bool, float [0..1], or 0..255 ints)
        if img.dtype == np.bool_:
            img_u8 = (img.astype(np.uint8) * 255)
        elif np.issubdtype(img.dtype, np.floating):
            img_u8 = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8) if img.max() <= 1.0 else np.clip(img, 0, 255).astype(np.uint8)
        else:
            img_u8 = np.clip(img, 0, 255).astype(np.uint8)

        _, binary = cv2.threshold(img_u8, 127, 255, cv2.THRESH_BINARY)

        # cv2.findContours has different return signatures across versions.
        # RETR_TREE keeps both outer contours and hole contours.
        cnts = cv2.findContours(binary.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours_i = cnts[0] if len(cnts) == 2 else cnts[1]

        # return all contours for this image as a list of (K,2) arrays
        contours_per_image = []
        for c in contours_i:
            pts = c.squeeze()
            if pts.ndim == 1:
                pts = pts.reshape(1, -1)
            contours_per_image.append(pts)

        # if no contours found, append an empty list
        if len(contours_per_image) == 0:
            contours.append([])
        else:
            contours.append(contours_per_image)

        # if n = 0 return all contours, else return the n largest contours by area
        if n > 0 and len(contours_per_image) > n:
            contours_per_image.sort(key=cv2.contourArea, reverse=True)
            contours[i] = contours_per_image[:n]

    return contours



def translation_invariant(features):
    """
    Make input Fourier descriptors invariant to translation.

    Args
    ----
    features: np.ndarray (N, D)
        The Fourier descriptors of N images over D features.

    Return
    ------
    features_inv: np.ndarray (N, K)
        The Fourier descriptors invariant to translation of N images 
        over K (K <= N) features.
    """

    # Set default values
    features_inv = np.zeros_like(features)
    
    features_inv = features.copy()

    # Translation in spatial domain only affects the DC term (k=0) in Fourier domain.
    # Remove it to make descriptors translation-invariant.
    if features_inv.ndim == 1:
        features_inv[0] = 0
    else:
        features_inv[:, 0] = 0
    
    return features_inv


def rotation_invariant(features):
    """
    Make input Fourier descriptors invariant to rotation.

    Args
    ----
    features: np.ndarray (N, D)
        The Fourier descriptors of N images over D features.

    Return
    ------
    features_inv: np.ndarray (N, K)
        The Fourier descriptors invariant to rotation of N images 
        over K (K <= N) features.
    """

    # Set default values
    features_inv = np.zeros_like(features)
    
    features_inv = features.copy()

    features_inv = np.abs(features)


    return features_inv


def scaling_invariant(features):
    """
    Make input Fourier descriptors invariant to scaling.

    Args
    ----
    features: np.ndarray (N, D)
        The Fourier descriptors of N images over D features.

    Return
    ------
    features_inv: np.ndarray (N, K)
        The Fourier descriptors invariant to scaling of N images 
        over K (K <= N) features.
    """

    # Set default values
    features_inv = np.zeros_like(features)
    
    features_inv = features.copy().astype(np.complex128)

    features_inv = features_inv / np.abs(features_inv[:, 1:2])
    
    return features_inv


def compute_distance_map(pattern: np.ndarray):
    """
    Compute the distance map for the given pattern. The values of the map are computed as 
    the distance to the closest pattern contour.

    Args
    ----
    pattern: np.ndarray (28, 28)
        Pattern to process

    Return
    ------
    distance_map: np.ndarray (28, 28)
        Distance map where each entry is the distance to the closest pattern contour (shortest 
        distance to pattern)
    """
    
    # Initialize dummy values
    distance_map = np.zeros_like(pattern)
    
    
    # ------------------
    pattern_bin = (pattern > 0).astype(np.uint8)
    contours, _ = cv2.findContours(pattern_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return distance_map
    contour_points = np.vstack([cnt.squeeze(axis=1) for cnt in contours])
    h, w = pattern.shape
    pixel_points = np.array([[x, y] for y in range(h) for x in range(w)])
    dists = euclidean_distances(pixel_points, contour_points)
    min_dists = np.min(dists, axis=1)
    distance_map = min_dists.reshape(h, w).astype(np.float32)
    # ------------------
    
    return distance_map


def compute_distance(imgs, d_map):
    """
    Compute the distances for each image with respect to the reference pattern using the precomputed 
    distance map. The final distance is the average of all distances from the image's contour points 
    to the reference pattern.

    Args
    ----
    imgs: np.ndarray (N, 28, 28)
        Source images
    d_map: np.ndarray (28, 28)
        The precomputed distance map where each entry is the distance to the closest pattern contour 
        (shortest distance to pattern)
    
    Return
    ------
    dist: np.ndarray (N, )
        Averaged distance to pattern for each input image.
    """
    
    # Default values
    dist = np.zeros(len(imgs))

    for i, img in enumerate(imgs):
        img_bin = (img > 0).astype(np.uint8)
        contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            dist[i] = 0.0
            continue
        contour_points = np.vstack([cnt.squeeze(axis=1) for cnt in contours])
        xs = contour_points[:, 0]
        ys = contour_points[:, 1]
        dist[i] = np.mean(d_map[ys, xs])
    
    return dist
    