import tensorflow as tf
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt

def load_img(path_to_img):
    """Loads and preprocesses an image from a given path."""
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
