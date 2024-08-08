import cv2
import numpy as np
import os
import urllib.request
import keras
import  tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import SVC
from keras_facenet import FaceNet



def download_file(url, filename):
    if not os.path.exists(filename):
        print(f'Downloading {filename}...')
        urllib.request.urlretrieve(url, filename)
        print(f'{filename} downloaded.')

download_file(caffe_model_url, caffe_model_file)
download_file(caffe_weights_url, caffe_weights_file)

facenet_model = FaceNet()

net = cv2.dnn.readNetFromCaffe(caffe_model_file, caffe_weights_file)
def get_embedding(model, face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = np.expand_dims(face_pixels, axis=0)
    yhat = model.embeddings(samples)
    return yhat[0]


def detect_faces(image):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = image[startY:endY, startX:endX]
            face = cv2.resize(face, (160, 160))
            return face
    return None

def load_faces(directory):
    faces = []
    labels = []
    #load faces funcition dataset me jake subdir me jake data likalega aur use facenet  me bhejega
    for subdir in os.listdir(directory):
        path = os.path.join(directory, subdir)
        if not os.path.isdir(path):
            continue
        for filename in os.listdir(path):
            file_path = 'C:\\Users\\Dell\\OpenCV\\' +  path + '\\' + filename
            print(os.path.exists(file_path))
            image = cv2.imread(file_path)
            face = detect_faces(image)
            if face is not None:
                #facenet embedings return krega (embedgings maine facial features..)
                embedding = get_embedding(facenet_model, face)
                faces.append(embedding)
                labels.append(subdir)
    return np.array(faces), np.array(labels)

def main():
    #yahan se  dataset ko accedss krega..
    dataset_path = 'dataset/'  
    faces, labels = load_faces(dataset_path)

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    normalizer = Normalizer(norm='l2')
    faces = normalizer.transform(faces)

    #facenet se mila embeddings svm classifier me jayega.. svm classifier embedings ko process krega to identify all images to individual person...
    classifier = SVC(kernel='linear', probability=True)
    classifier.fit(faces, labels)

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        #yahan se realtime face capture start hoga
        face = detect_faces(frame)
        if face is not None:
            #aur call krega to get processed embedings
            face_embedding = get_embedding(facenet_model, face)
            face_embedding = normalizer.transform([face_embedding])
            #data ke basis pe recognise krega aur probability generate krega..
            predictions = classifier.predict_proba(face_embedding)
            class_index = np.argmax(predictions)
            class_probability = predictions[0, class_index]
            predicted_label = label_encoder.inverse_transform([class_index])

            if class_probability > 0.3:  
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
