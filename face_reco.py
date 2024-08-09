import os
from os import listdir
from numpy import asarray, expand_dims
import pickle
import cv2
from keras_facenet import FaceNet
from mtcnn import MTCNN
from sklearn.metrics.pairwise import cosine_similarity

#mtcnn ka use kiya h detectino k liye caffee was waste
detector = MTCNN()
MyFaceNet = FaceNet()

#dataset ka pth
folder = 'dataset/'
database_file = "data.pkl"
database = {}

def save_database():
    with open(database_file, "wb") as myfile:
        pickle.dump(database, myfile)

def load_database():
    global database
    #agar data.pkl present h to thik othw train new data
    if os.path.exists(database_file):
        with open(database_file, "rb") as myfile:
            database = pickle.load(myfile)
        print("Database loaded from file.")
    else:
        for filename in listdir(folder):
            for file in listdir(f'C://Users//Dell//OpenCV//dataset//{filename}'):
                vid = cv2.imread(f'C://Users//Dell//OpenCV//dataset//{filename}//{file}')

                #yahan pe dtection strt hogi
                faces = detector.detect_faces(vid)

                for face_data in faces:
                    x1, y1, width, height = face_data['box']
                    x1, y1 = abs(x1), abs(y1)
                    x2, y2 = x1 + width, y1 + height

                    # cropping faces kyoki facenet needs 160 by 160
                    face = vid[y1:y2, x1:x2]
                    face = cv2.resize(face, (160, 160))

                    # face ko convert krna pdega array me, normalization ke liye
                    face = asarray(face)
                    face = expand_dims(face, axis=0)

                    #embedding generate hogi
                    signature = MyFaceNet.embeddings(face)

                    #embeddings data.pkl me save hojaygi
                    database[os.path.splitext(filename)[0]] = signature

        save_database()

load_database()

#ye screen capture k liye
cap = cv2.VideoCapture(0)

while True:
    _, vid = cap.read()
    
    #call detector
    faces = detector.detect_faces(vid)

    for face_data in faces:
        x1, y1, width, height = face_data['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height

        #aagain crop and resize fce 
        face = vid[y1:y2, x1:x2]
        face = cv2.resize(face, (160, 160))

        # normalization
        face = asarray(face)
        face = expand_dims(face, axis=0)

        # facenet embedding genearte kro 
        signature = MyFaceNet.embeddings(face)

        max_similarity = -1  
        identity = 'Unknown'
        for key, value in database.items():
            #is version me humlog cosine similarity bhi check krenge, that will improve accuracy a lot
            similarity = cosine_similarity(value, signature)[0][0]
            if similarity > max_similarity:
                max_similarity = similarity
                identity = key

        if max_similarity > 0.6:  #threshold valu
            cv2.putText(vid, identity, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(vid, 'Unknown', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.rectangle(vid, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('res', vid)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
