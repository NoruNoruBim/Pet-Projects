import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from tensorflow import keras
import numpy as np


def viewImage(image, name="Display"):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#   make prediction for one symbol
def cnn_digits_predict(model, symbol):
    image = cv2.imread(symbol)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.resize(image, (40, 40), interpolation = cv2.INTER_AREA)
    image = image / 255.
    image = image.reshape((1, 40, 40, 1))

    return np.argmax(model.predict([image]), axis=1)


#   cut symbols from licence plate image and save into folder
def extract(name, show=False):
    orig = cv2.imread(name)

    img = orig.copy()

    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_image, 40, 80, apertureSize = 3)
    edges = cv2.dilate(edges, np.ones((1, 1), np.uint8), iterations=1)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    candidates = []
    for contour in contours:
        tmp = cv2.boundingRect(contour)
        if tmp[2] * tmp[3] < len(img) * len(img[0]) * 0.2 and\
           tmp[2] * tmp[3] > len(img) * len(img[0]) * 0.025 and\
           tmp[3] / tmp[2] > 1:
            candidates += [tmp]
    if show:
        print("candidates:", len(candidates))

    output = img.copy()
    for i in range(len(candidates)):
        x, y, w, h = candidates[i]
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)

    good = []
    for i, c in enumerate(candidates):
        tmp = True
        for j, c1 in enumerate(candidates):
            if i != j and c1[0] < (c[0] + c[0] + c[2]) // 2 < c1[0] + c1[2] and c[2] * c[3] < c1[2] * c1[3]:
                tmp = False
                break
        if tmp:
            good += [c]
        
    good = list(set(good))

    if show:
        print("good:", len(good))
    good.sort(key=lambda x: x[0], reverse=False)

    for l in os.listdir("my_symbols"):
        os.remove("my_symbols/" + l)

    for i in range(0, len(good), 1):
        x, y, w, h = good[i]
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
        symbol = img[max(0, y - int(h * 0.3)) : min(len(img), y + int(h * 1.1)),\
                     max(0, x - int(w * 0.2)) : min(len(img[0]), x + int(w * 1.2))]
        cv2.imwrite("my_symbols/symbol" + str(i + 1) + ".png", symbol)

    if show:
        cv2.imshow("0", img)
        cv2.imshow("1", gray_image)
        cv2.imshow("2", edges)
        cv2.imshow("4", output)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


#   little improvement
def make_better(labels):
    if len(labels) >= 1:
        if labels[0] == '0':
            labels[0] = 'O'
        if labels[0] == '8':
            labels[0] = 'B'

    if len(labels) >= 5:
        if labels[4] == '0':
            labels[4] = 'O'
        if labels[4] == '8':
            labels[4] = 'B'
    
    if len(labels) >= 6:
        if labels[5] == '0':
            labels[5] = 'O'
        if labels[5] == '8':
            labels[5] = 'B'
    
    if len(labels) == 9:
        if labels[-3] == '7' and labels[-2] == '7':
            labels[-1] = '7'
        elif labels[-1] == '7':
            labels[-3] = '7'
            labels[-2] = '7'
        if labels[-3] != '7' and labels[-3] != '1' and labels[-3] != '?':
            labels = labels[:-1]

    return ''.join(labels)


def main(name="box.png", show=False):
    symbol_to_number = {0 : '0', 1 : '1', 2 : '2', 3 : '3', 4 : '4', 5 : '5', 6 : '6', 7 : '7', 8 : '8', 9 : '9',\
           10 : 'A', 11 : 'B', 12 : 'C', 13 : 'E', 14 : 'H', 15 : 'K', 16 : 'M', 17 : '!', 18 : 'P', 19 : 'T', 20 : 'X', 21 : 'Y', 22 : '?'}

    #   load model and weights (deserialize)
    model = keras.models.load_model('cnn1.h5')
    
    #   extract symbols from plate
    extract(name, show)

    #   make prediction for each symbol
    labels = []
    for i in os.listdir("my_symbols"):
        labels += [symbol_to_number[int(cnn_digits_predict(model, "my_symbols/" + i)[0])]]

    return labels
