import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import time


def cnn_model():
    image_size = 40
    num_channels = 1  # 1 for grayscale images
    num_classes = 23  # Number of outputs
    
    model = Sequential()
    
    model.add(Conv2D(filters=32,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same',
                     input_shape=(image_size, image_size, num_channels)))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    
    # Densely connected layers
    model.add(Dense(128, activation='relu'))
    
    # Output layer
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])# Adam - better if compare with others

    return model


def cnn_train(model):
    X, y = get_base()
    print(len(y), len(X))

    idx = np.random.permutation(X.shape[0])
    
    X, y = X[idx], y[idx]

    X_train = X[:int(len(X) * 0.8)]
    y_train = y[:int(len(X) * 0.8)]

    X_test = X[int(len(X) * 0.8):]
    y_test = y[int(len(X) * 0.8):]

    # Get image size
    image_size = 40
    num_channels = 1  # 1 for grayscale images

    # re-shape and re-scale the images data
    train_data = np.reshape(X_train, (X_train.shape[0], image_size, image_size, num_channels))
    train_data = train_data.astype('float32') / 255.0
    
    # encode the labels - we have 10 output classes
    # 3 -> [0 0 0 1 0 0 0 0 0 0], 5 -> [0 0 0 0 0 1 0 0 0 0]
    num_classes = 23
    print(y_train.shape)
    y_train_cat = keras.utils.to_categorical(y_train, num_classes)

    # re-shape and re-scale the images validation data
    val_data = np.reshape(X_test, (X_test.shape[0], image_size, image_size, num_channels))
    val_data = val_data.astype('float32') / 255.0
    
    # encode the labels - we have 10 output classes
    y_test_cat = keras.utils.to_categorical(y_test, num_classes)

    print("Training the network...")
    t_start = time.time()

    # Start training the network
    model.fit(train_data, y_train_cat, epochs=60, batch_size=64, validation_data=(val_data, y_test_cat))

    print("Done, dT:", time.time() - t_start)

    return model, train_data, y_train_cat, val_data, y_test_cat


def get_base():
    X = []
    y = []
    
    path = r"symbols/"
    
    for i in range(23):
        if i != 17:
            for f in os.listdir(path + str(i)):
                y += [i]
                tmp = cv2.imread(path + str(i) + '/' + f)
                tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
                tmp = cv2.resize(tmp, (40, 40), interpolation = cv2.INTER_AREA)
                X += [tmp]
    return np.array(X), np.array(y)


model = cnn_model()
model, x_train, y_train, x_test, y_test = cnn_train(model)
model.save('cnn1.h5')

y_train_pred = np.array([np.argmax(x, axis=0) for x in model.predict(x_train)])
y_train = np.array([np.argmax(x, axis=0) for x in y_train])
train_acc = np.sum(y_train == y_train_pred, axis=0) / x_train.shape[0]

y_test_pred = np.array([np.argmax(x, axis=0) for x in model.predict(x_test)])
y_test = np.array([np.argmax(x, axis=0) for x in y_test])
test_acc = np.sum(y_test == y_test_pred, axis=0) / x_test.shape[0]

print("Train acc: %.3f" % train_acc)
print("Test acc: %.3f" % test_acc)
