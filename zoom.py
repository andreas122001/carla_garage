import scipy.ndimage
import numpy as np
import cv2


a = np.arange(0, 36).reshape(6,6)

from scipy.ndimage import zoom
from skimage.transform import resize  # You can use other libraries for resizing as well.

def zoom_and_resize(array, zoom_factor):
    # Step 1: Zoom in
    new_array = zoom(array, zoom=zoom_factor, order=3)  # Order=3 is cubic interpolation.
    
    # Step 2: Resize back to original shape
    original_shape = array.shape
    # new_array = resize(new_array, original_shape, mode='reflect', anti_aliasing=True)
    
    return new_array

def zoom_at(img, zoom, coord=None):
    """
    Simple image zooming without boundary checking.
    Centered at "coord", if given, else the image center.

    img: numpy.ndarray of shape (h,w,:)
    zoom: float
    coord: (float, float)
    """
    # Translate to zoomed coordinates
    cx, cy = np.round((img.shape[0]) / 2), np.round((img.shape[1]) / 2)
    h, w = np.round(img.shape[0] * zoom), np.round(img.shape[1] * zoom)


    img = img[int(cx-w):int(cx+w), int(cy-h):int(cy+h)]

    
    return img

print(a)
# b = scipy.ndimage.interpolation.zoom(a, 0.6, order=0)
# b = zoom_at(a, 0.4)
# b = np.resize(b, (5, 5))
b = zoom_and_resize(a, 0.5)

print(b)
print(b.shape)


