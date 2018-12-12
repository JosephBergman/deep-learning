import scipy
import numpy as np
from keras import backend as K 
from keras.applications import inception_v3
from keras.preprocessing import image

def resize_image(img, size):
    img = np.copy(img)
    factors = (1, float(size[0]) / img.shape[1], float(size[1]) / img.shape[2], 1) 
    return scipy.ndimage.zoom(img, factors, order=1)

def save_image(img, filename):
    pil_image = deprocess_image(np.copy(img))
    scipy.misc.imsave(filename, pil_image)
    
def preprocess_image(image_path):
    img = image.load_img(image_path)
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img

def deprocess_image(x):
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((x.shape[1], x.shape[2], 3))
    x /= 2.
    x += 0.5
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def image_octaves(img, octave_scale, num_octaves):
    original_shape = img.shape[1:3]
    successive_shapes = [original_shape]
    for i in range(1, num_octaves):
        shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
        successive_shapes.append(shape)
    return successive_shapes[::-1]