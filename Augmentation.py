import cv2
import numpy as np
from numpy import expand_dims
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageEnhance

# Note : Must take colored image
img2 = cv2.imread("5.jpg")

img = Image.open("5.jpg")
enhancer = ImageEnhance.Contrast(img)
img = enhancer.enhance(1.5)
img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)

cv2.imshow('contrasted', img)
cv2.waitKey(0)


def Rotation(img):
    Change_Dimension = expand_dims(img, 0)
    Rotate = ImageDataGenerator(rotation_range=50, fill_mode='nearest')
    it = Rotate.flow(Change_Dimension, batch_size=1)
    Next_it = it.next()
    Rotated_Image = Next_it[0].astype('uint8')
    cv2.imshow("image", Rotated_Image)
    cv2.waitKey(0)
    return Rotated_Image


# Rotation(img)


def Width_Shift(img):
    Change_Dimension = expand_dims(img, 0)
    Shifting = ImageDataGenerator(width_shift_range=0.4, fill_mode='nearest')
    it = Shifting.flow(Change_Dimension, batch_size=1)
    Next_it = it.next()
    Shifted_Image = Next_it[0].astype('uint8')
    cv2.imshow("image", Shifted_Image)
    cv2.waitKey(0)
    return Shifted_Image


# Width_Shift(img)


def Height_Shift(img):
    Change_Dimension = expand_dims(img, 0)
    Shifting = ImageDataGenerator(height_shift_range=0.4, fill_mode='nearest')
    it = Shifting.flow(Change_Dimension, batch_size=1)
    Next_it = it.next()
    Shifted_Image = Next_it[0].astype('uint8')
    cv2.imshow("image", Shifted_Image)
    cv2.waitKey(0)
    return Shifted_Image


# Height_Shift(img)


def Zoom(img):
    Change_Dimension = expand_dims(img, 0)
    Zooming = ImageDataGenerator(zoom_range=0.5)
    it = Zooming.flow(Change_Dimension, batch_size=1)
    Next_it = it.next()
    Zoomed_Image = Next_it[0].astype('uint8')
    cv2.imshow("image", Zoomed_Image)
    cv2.waitKey(0)
    return Zoomed_Image


# Zoom(img)


def Horizontal_Flip(img):
    Change_Dimension = expand_dims(img, 0)
    Flipping = ImageDataGenerator(horizontal_flip=True, fill_mode='nearest', )
    it = Flipping.flow(Change_Dimension, batch_size=1)
    Next_it = it.next()
    Flipped_Image = Next_it[0].astype('uint8')
    cv2.imshow("image", Flipped_Image)
    cv2.waitKey(0)
    return Flipped_Image


# Horizontal_Flip(img)


def Vertical_Flip(img):
    Change_Dimension = expand_dims(img, 0)
    Flipping = ImageDataGenerator(vertical_flip=True, fill_mode='nearest')
    it = Flipping.flow(Change_Dimension, batch_size=1)
    Next_it = it.next()
    Flipped_Image = Next_it[0].astype('uint8')
    cv2.imshow("image", Flipped_Image)
    cv2.waitKey(0)
    return Flipped_Image


def Change_Brightness(img):
    Change_Dimension = expand_dims(img, 0)
    New_Brightness = ImageDataGenerator(brightness_range=[0.4, 1.3])
    it = New_Brightness.flow(Change_Dimension, batch_size=1)
    Next_it = it.next()
    Image = Next_it[0].astype('uint8')
    cv2.imshow("image", Image)
    cv2.waitKey(0)
    return Image


# Change_Brightness(img)
