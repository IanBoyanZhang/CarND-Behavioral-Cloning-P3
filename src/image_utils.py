import cv2
import numpy as np
from config import W, H

def to_rgb(image):
    """
    cv2.imread default: bgr
    """
    return cv2.cvtColor(image, cv2.BGR2RGB)

def crop_image(image, ROI=(65, 125), size=(W, H)):
    """
    Restrict ROI
    """
    return cv2.resize(image[ROI[0]: ROI[1]], size, cv2.INTER_AREA)

def expand_image(image, size=(320, 160)):
    """
    """
    return cv2.resize(image, size, cv2.INTER_AREA)

def to_255(image):
    return np.floor((image + 1)*127.5)

def flip_image(image):
    return cv2.flip(image, 1)

def translate_image(image, tx):
    """
    Affine/translation only
    http://docs.opencv.org/trunk/da/d6e/tutorial_py_geometric_transformations.html
    """
    rows = image.shape[0]
    cols = image.shape[1]
    M = np.float32([[1, 0, tx], [0, 1, 0]])
    return cv2.warpAffine(image, M, (cols, rows))

def color_select(image):
    """
    Only takes S channel
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[:, :, 1]

def stack_image_planes(image):
    """
    Assume input image is 2D x 1ch
    Output: 2D x 3ch
    """
    stack = image + 1
    return np.dstack((stack, stack, stack))*255

def resize_image(image):
    return cv2.resize(image, (320, 160), cv2.INTER_AREA)

def preprocess(image):
    """
    Perform your pre-processing EXACTLY the same as you do it in model.py
    Typically this would just be things like normalization and image resizing
    Directly put those in your model?
    """
    return crop_image(color_select(image))

def interim_process(image):
    """
    Visualizae intermediate layers
    """
    return stack_image_planes(resize_image(image))

def process_image_ex(image):
    """
    """
    NotImplementedError


