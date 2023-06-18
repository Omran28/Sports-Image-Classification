# Imports
import os
import numpy as np
from PIL import Image, ImageEnhance
import cv2
import glob
import csv
from random import shuffle
from numpy import expand_dims
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.framework import ops
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression


# Augmentation methods
def Horizontal_Flip(image_Augmented):
    Change_Dimension = expand_dims(image_Augmented, 0)
    Flipping = ImageDataGenerator(horizontal_flip=True, fill_mode='nearest', )
    it = Flipping.flow(Change_Dimension, batch_size=1)
    Next_it = it.next()
    Flipped_Image = Next_it[0].astype('uint8')
    return Flipped_Image


def Change_Brightness(image_Augmented):
    Change_Dimension = expand_dims(image_Augmented, 0)
    New_Brightness = ImageDataGenerator(brightness_range=[0.4, 0.4])
    it = New_Brightness.flow(Change_Dimension, batch_size=1)
    Next_it = it.next()
    image_Augmented = Next_it[0].astype('uint8')
    return image_Augmented


def Rotation(image_Augmented):
    Change_Dimension = expand_dims(image_Augmented, 0)
    Rotate = ImageDataGenerator(rotation_range=50, fill_mode='nearest')
    it = Rotate.flow(Change_Dimension, batch_size=1)
    Next_it = it.next()
    Rotated_Image = Next_it[0].astype('uint8')
    return Rotated_Image


def Zoom(image_Augmented):
    Change_Dimension = expand_dims(image_Augmented, 0)
    Zooming = ImageDataGenerator(zoom_range=0.5)
    it = Zooming.flow(Change_Dimension, batch_size=1)
    Next_it = it.next()
    Zoomed_Image = Next_it[0].astype('uint8')
    return Zoomed_Image


# Min-Max Normalization
def Rescale(image_Augmented, min_Value, max_Value_Aug):
    image_Augmented = (image_Augmented - image_Augmented.min()) / float(image_Augmented.max() - image_Augmented.min())
    image_Augmented = min_Value + image_Augmented * (max_Value_Aug - min_Value)
    return image_Augmented


# Augmentation
def Augmentation(images, label):
    new_images = []
    images_label = []
    for aug_Image in images:
        # original image
        images_label.append(label)

        # flip left to right
        L_R_flipped = Horizontal_Flip(aug_Image)
        new_images.append(L_R_flipped)
        images_label.append(label)

        # combination of flip L to R & Brightness
        # brightened_Flipped = Change_Brightness(L_R_flipped)
        # new_images.append(brightened_Flipped)
        # images_label.append(label)

        # zoom
        zoomed = Zoom(aug_Image)
        new_images.append(zoomed)
        images_label.append(label)

        # # rotation
        # rotated = Rotation(img)
        # new_images.append(rotated)
        # images_label.append(label)

        # change the brightness
        brightened = Change_Brightness(aug_Image)
        new_images.append(brightened)
        images_label.append(label)

    images += new_images
    shuffle(images)
    return images, images_label


# Create Train and Validation data
def Create_Training_Data(b, f, r, s, t, y):
    b, B_label = Augmentation(b, [1, 0, 0, 0, 0, 0])
    f, F_label = Augmentation(f, [0, 1, 0, 0, 0, 0])
    r, R_label = Augmentation(r, [0, 0, 1, 0, 0, 0])
    s, S_label = Augmentation(s, [0, 0, 0, 1, 0, 0])
    t, T_label = Augmentation(t, [0, 0, 0, 0, 1, 0])
    y, Y_label = Augmentation(y, [0, 0, 0, 0, 0, 1])

    # Training data
    training_Data = []
    train = b[0:592] + f[0:592] + r[0:592] + s[0:592] + t[0:592] + y[0:592]
    train_labels = B_label[0:592] + F_label[0:592] + R_label[0:592] + S_label[0:592] + T_label[0:592] + Y_label[0:592]

    # validation data
    Validation_data = []
    validation = b[592:740] + f[592:740] + r[592:740] + s[592:740] + t[592:740] + y[592:740]
    validation_labels = B_label[0:148] + F_label[0:148] + R_label[0:148] + S_label[0:148] + T_label[0:148] + Y_label[
                                                                                                   0:148]

    # Total 4440, 3552 train & 888 validation
    for i in range(len(train)):
        final_Image = cv2.resize(train[i], (size, size))
        final_Image = Rescale(final_Image, 0, 1)
        training_Data.append([final_Image, np.array(train_labels[i])])
        if i < 888:
            final_Image = cv2.resize(validation[i], (size, size))
            final_Image = Rescale(final_Image, 0, 1)
            Validation_data.append([final_Image, np.array(validation_labels[i])])

    shuffle(training_Data)
    shuffle(Validation_data)
    return training_Data, Validation_data


# Architecture
def Model_Architecture():
    ops.reset_default_graph()
    conv_input = input_data(shape=[None, size, size, 3], name='input')
    conv1 = conv_2d(conv_input, 64, 5, activation='relu')  # 100 * 100
    pool1 = max_pool_2d(conv1, 5)

    conv2 = conv_2d(pool1, 128, 5, activation='relu')
    pool2 = max_pool_2d(conv2, 5)

    conv3 = conv_2d(pool2, 64, 5, activation='relu')
    pool3 = max_pool_2d(conv3, 5)

    # conv4 = conv_2d(pool3, 128, 5, activation='relu')
    # pool4 = max_pool_2d(conv4, 5)
    #
    # conv5 = conv_2d(pool4, 64, 5, activation='relu')
    # pool5 = max_pool_2d(conv5, 5)

    fully_layer = fully_connected(pool3, 1024, activation='relu')
    fully_layer = fully_connected(fully_layer, 1024, activation='relu')
    fully_layer = dropout(fully_layer, 0.5)

    cnn_layers = fully_connected(fully_layer, 6, activation='softmax')

    cnn_layers = regression(cnn_layers, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy',
                            name='targets')
    Model = tflearn.DNN(cnn_layers, tensorboard_dir='log', tensorboard_verbose=3)
    return Model


# Parameters
size = 100
LR = 0.001
MODEL_NAME = "Sports Classification"

# Load Train Data
train_images_path = glob.glob("Train\*.*")

# Read train images and separate each sport
B = []  # 196
F = []  # 400
R = []  # 202
S = []  # 240
T = []  # 185
Y = []  # 458
for img in train_images_path:
    if img[6] == 'B':
        # img = cv2.imread(img)
        img = Image.open(img)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        B.append(img)
    elif img[6] == 'F':
        # img = cv2.imread(img)
        img = Image.open(img)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        F.append(img)
    elif img[6] == 'R':
        # img = cv2.imread(img)
        img = Image.open(img)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        R.append(img)
    elif img[6] == 'S':
        # img = cv2.imread(img)
        img = Image.open(img)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        S.append(img)
    elif img[6] == 'T':
        # img = cv2.imread(img)
        img = Image.open(img)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        T.append(img)
    elif img[6] == 'Y':
        # img = cv2.imread(img)
        img = Image.open(img)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        Y.append(img)

# Load Test Data
test_images_path = glob.glob("Test/*.*")
if os.path.exists('testing_data.npy'):
    test_images = np.load('testing_data.npy', allow_pickle=True)
else:
    # reading and resizing
    test_images = []
    for testing_image in test_images_path:
        # img = cv2.imread(testing_image)
        img = Image.open(testing_image)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size))
        image = Rescale(img, 0, 1)
        test_images.append(img)
    np.save('testing_data.npy', test_images)

# Split data into Train and Test
if os.path.exists('training_data.npy'):
    training_data = np.load('training_data.npy', allow_pickle=True)
    validation_data = np.load('validation_data.npy', allow_pickle=True)
else:
    training_data, validation_data = Create_Training_Data(B, F, R, S, T, Y)
    np.save('training_data.npy', training_data)
    np.save('validation_data.npy', validation_data)

# Training data
x_train = np.array([i[0] for i in training_data]).reshape(-1, size, size, 3)
y_train = [i[1] for i in training_data]

# Validation data
x_validate = np.array([i[0] for i in validation_data]).reshape(-1, size, size, 3)
y_validate = [i[1] for i in validation_data]

# Model Training
model = Model_Architecture()
if os.path.exists('model.tfl.meta'):
    model.load('./model.tfl')
else:
    model.fit({'input': x_train}, {'targets': y_train}, n_epoch=8,
              validation_set=({'input': x_validate}, {'targets': y_validate}),
              snapshot_step=500, show_metric=True, run_id=MODEL_NAME)
    model.save('model.tfl')

# Model Testing
with open('classification.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['image_name', 'label'])
    for i in range(len(test_images)):
        tested_image = test_images[i].reshape(-1, size, size, 3)
        prediction = model.predict(tested_image)[0]
        max_Value = 0.0
        index = 0
        for j in range(len(prediction)):
            if prediction[j] > max_Value:
                max_Value = prediction[j]
                index = j
        img_name = test_images_path[i].split("\\")[1]
        writer.writerow([str(img_name), index])
#         print(f"basketball: {prediction[0]}, football: {prediction[1]}, Rowing: {prediction[2]} ,"
#               f"swimming: {prediction[3]}, "f"tennis: {prediction[4]}, yoga: {prediction[5]}")
