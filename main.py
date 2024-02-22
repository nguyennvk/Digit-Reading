import os
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(10, activation='softmax'))
#
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(x_train, y_train, epochs=99)
# model.save('digit_reading.model')

model = tf.keras.models.load_model('digit_reading.model')


def margin(image):
    r, c = image.shape[0], image.shape[1]
    start = 1
    l = []
    white = True
    while start < c:
        for x in range(start, c):
            check = is_white_column(image, x)
            if white != check:
                white = check
                l.append(x)

            start+=1
    return l
def is_white_column(image, r):
    for x in range(image.shape[0]):
        if image[x, r] != 255:
            return False
    return True


if __name__ == "__main__":
    path = input("Enter the file path: ")
    img = cv2.imread(path)[:, :, 0]
    result = ""
    l = margin(img)
    width, height = 28, 28
    for x in range(0, len(l), 2):
        img1 = img[0:img.shape[1], l[x]-3:l[x+1]+3]
        img1 = cv2.resize(img1, (width, height), interpolation=cv2.INTER_AREA)
        img1 = np.invert(np.array([img1]))
        prediction = model.predict(img1)
        result += str(np.argmax(prediction))
    print(f"The prediction is {result}")

