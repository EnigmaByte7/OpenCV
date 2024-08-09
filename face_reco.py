import cv2
import numpy as np
import os
import urllib.request
import keras
import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import SVC
from keras_facenet import FaceNet
from mtcnn import MTCNN  # MTCNN import kiya

# Caffe model files ko download karne ke liye URLs
caffe_model_url = 'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt'
caffe_model_file = 'deploy.prototxt'
caffe_weights_url = 'https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel'
caffe_weights_file = 'res10_300x300_ssd_iter_140000.caffemodel'

def download_file(url, filename):
    if not os.path.exists(filename):
        print(f'Downloading {filename}...')
        urllib.request.urlretrieve(url, filename)
        print(f'{filename} downloaded.')

download_file(caffe_model_url, caffe_model_file)
download_file(caffe_weights_url, caffe_weights_file)

facenet_model = FaceNet()

# MTCNN ka instance create kiya
mtcnn = MTCNN()

def get_embedding(model, face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = np.expand_dims(face_pixels, axis=0)
    yhat = model.embeddings(samples)
    return yhat[0]

def detect_faces(image):
    # MTCNN se face detect karne ka kaam
    faces = mtcnn.detect_faces(image)
    if faces:
        x, y, width, height = faces[0]['box']
        x, y = abs(x), abs(y)
        face = image[y:y+height, x:x+width]
        face = cv2.resize(face, (160, 160))
        return face
    return None

def load_faces(directory):
    faces = []
    labels = []
    # load faces funcition dataset me jake subdir me jake data likalega aur use facenet  me bhejega
    for subdir in os.listdir(directory):
        path = os.path.join(directory, subdir)
        if not os.path.isdir(path):
            continue
        for filename in os.listdir(path):
            print(filename)
            file_path = 'C:\\Users\\Dell\\OpenCV\\' +  path + '\\' + filename
            image = cv2.imread(file_path)
            face = detect_faces(image)
            if face is not None:
                # facenet embedings return krega (embedgings maine facial features..)
                embedding = get_embedding(facenet_model, face)
                faces.append(embedding)
                labels.append(subdir)
    print(labels)
    return np.array(faces), np.array(labels)

def main():
    # yahan se dataset ko accedss krega..
    dataset_path = 'dataset/'  
    faces, labels = load_faces(dataset_path)

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    normalizer = Normalizer(norm='l2')
    faces = normalizer.transform(faces)

    # facenet se mila embeddings svm classifier me jayega.. svm classifier embedings ko process krega to identify all images to individual person...
    classifier = SVC(kernel='linear', probability=True)
    classifier.fit(faces, labels)

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # yahan se realtime face capture start hoga
        face = detect_faces(frame)
        if face is not None:
            # aur call krega to get processed embedings
            face_embedding = get_embedding(facenet_model, face)
            face_embedding = normalizer.transform([face_embedding])
            # data ke basis pe recognise krega aur probability generate krega..
            predictions = classifier.predict_proba(face_embedding)
            class_index = np.argmax(predictions)
            class_probability = predictions[0, class_index]
            predicted_label = label_encoder.inverse_transform([class_index])

            if class_probability > 0.5:  
                text = f'{predicted_label[0]} ({class_probability*100:.2f}%)'
            else:
                text = 'Unknown'

            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
