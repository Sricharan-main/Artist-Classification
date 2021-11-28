import json
import joblib
import cv2
import numpy as np
import base64
from matplotlib import pyplot as plt
import pywt

__class_name_to_num = {}
__class_num_to_name = {}
__model = None


def w2d(img, mode='haar', level=1):
    imArray = img

    imArray = cv2.cvtColor(imArray, cv2.COLOR_RGB2GRAY)
    imArray = np.float32(imArray)

    imArray = imArray / 255
    # compute co-efficients
    coeffs = pywt.wavedec2(imArray, mode, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0
    # recontruct
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)
    return imArray_H


def get_cv2_image_from_base64_data(b64str):

    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def get_cropped(image_b64_data, file_path):

    if file_path:

        img = cv2.imread(file_path)

    else:

        img = get_cv2_image_from_base64_data(image_b64_data)

    face_cascade = cv2.CascadeClassifier(
        './opencv/haarcascade_frontalface_default.xml')

    eye_cascade = cv2.CascadeClassifier('./opencv/haarcascade_eye.xml')

    dets = []

    if img is None:

        return dets

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(img_gray, 1.3, 5)

    for (x, y, w, h) in faces:

        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        roi_gray = img_gray[y:y + h, x:x + w]

        roi_color = img[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)

        if len(eyes) >= 2:

            dets.append(roi_color)

    return dets


def class_number_to_name(class_num):

    return __class_num_to_name[class_num]


def classify_image(image_b64_data, file_path=None):

    sz = 64

    load_artifacts()

    imgs = get_cropped(image_b64_data, file_path)

    results = []

    for img in imgs:

        scaled_img = cv2.resize(img, (sz, sz))

        img_har = w2d(img, 'db1', 5)

        scaled_img_har = cv2.resize(img_har, (sz, sz))

        combined_img = np.vstack(
            (scaled_img.reshape(sz * sz * 3,
                                1), scaled_img_har.reshape(sz * sz, 1)))

        X = combined_img.reshape(1, len(combined_img))

        Y = __model.predict(X)[0]
        
        results.append({
            'class':
            class_number_to_name(Y),

            'class_probability':
            np.around(__model.predict_proba(X) * 100, 2).tolist()[0],

            'class_dictionary':
            __class_name_to_num
        })

    return results


def get_b64_image():

    with open('./test_images/ntr-b64.txt') as f:

        return f.read()


def load_artifacts():

    print("Loading artifacts .........begin")

    global __class_name_to_num

    global __class_num_to_name

    with open('./artifacts/class_dictionary.json', 'r') as f:

        __class_name_to_num = json.load(f)

        __class_num_to_name = {

            num: name

            for name, num in __class_name_to_num.items()

        }

    global __model

    if __model is None:

        with open('./artifacts/saved_model.pkl', 'rb') as f:

            __model = joblib.load(f)

    print("Loading artifacts is complete")


if __name__ == '__main__':

    print(classify_image(get_b64_image(), None))
