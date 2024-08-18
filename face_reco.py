import os
from os import listdir
from numpy import asarray, expand_dims
import pickle
import cv2
from keras_facenet import FaceNet
from mtcnn import MTCNN
from sklearn.metrics.pairwise import cosine_similarity
from tkinter import *
import csv
from PIL import ImageTk, Image
import shutil
from sklearn.metrics.pairwise import euclidean_distances

#mtcnn ka use kiya h detectino k liye caffee was waste
detector = MTCNN()
MyFaceNet = FaceNet()

#dataset ka pth is variable now, agar koi new data hai to others ke liye train kro 
folder = 'others/'

database_file = "data.pkl"
database = {}

#ye function data.pkl ka data dekhne ke liye hai
def viewpkl():
    if os.path.exists('data.pkl'):
        with open(database_file, "rb") as myfile:
            database = pickle.load(myfile)
            for name, emb in database.items():
                print("Name : ", name, "Embeddings : ", emb, end='\n')

def save_database():
    with open(database_file, "wb") as myfile:
        pickle.dump(database, myfile)

def load_database():
    global database
    #agar data.pkl present h to thik othw train new data
    if os.path.exists(database_file):
        with open(database_file, "rb") as myfile:
            database = pickle.load(myfile)
            if(database == {}):
                # agar somehow the data.pkl is empty
                trainer('dataset1/')
    else:
        trainer('dataset1/')
    print("Database loaded from file.")

def trainer(folder):
    for filename in listdir(f'C://Users//Dell//OpenCV//{folder}'):
        for file in listdir(f'C://Users//Dell//OpenCV//{folder}//{filename}'):
            vid = cv2.imread(f'C://Users//Dell//OpenCV//{folder}//{filename}//{file}')

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

    if(folder == 'others/'):
        print("Model Triained on the new dataset")
        destination = 'C://Users//Dell//OpenCV//dataset1//'
        main = 'C://Users//Dell//OpenCV//others//'
        for folders in os.listdir(main):
            shutil.move(f'C://Users//Dell//OpenCV//others//{folders}', f'C://Users//Dell//OpenCV//dataset1')
            print("Moved :", folders)
    else:
        print("Model trained on the big dataset")

    save_database()

load_database()
if os.listdir('C://Users//Dell//OpenCV//others') != {}:
    trainer('others/')

#viewpkl()

cap = cv2.VideoCapture(0)
#ye hai vo set jisme identified logo ke name save honge
s = set()

def video_capture():
    back = cv2.imread('C://Users//Dell//Downloads//Group 1.png')
    #ye screen capture k liye

    cv2.namedWindow('Video Feed', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Video Feed', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    while True:
        _, vid = cap.read()
        vid = cv2.resize(vid, (800,500))
        #call detector
        faces = detector.detect_faces(vid)

        for face_data in faces:
            x1, y1, width, height = face_data['box']
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height

            #aagain crop and resize fce 
            face = vid[y1:y2, x1:x2]
            face = cv2.resize(face, (600, 600))

            # normalization
            face = asarray(face)
            face = expand_dims(face, axis=0)

            # facenet embedding genearte kro 
            signature = MyFaceNet.embeddings(face)

            max_similarity = -1  
            min_euclidean_distance = float('inf')
            identity = 'Unknown'

            for key, value in database.items():
                # Cosine similarity check
                similarity = cosine_similarity(value, signature)[0][0]
                if similarity > max_similarity:
                    max_similarity = similarity
                    identity = key

                # Euclidean distance check
                euclidean_distance = euclidean_distances(value, signature)[0][0]
                if euclidean_distance < min_euclidean_distance:
                    min_euclidean_distance = euclidean_distance
                    identity = key

            # ab combined value se threshold check kro
            if max_similarity > 0.6 and min_euclidean_distance < 10.0:  # eucludean threshold
                cv2.putText(vid, identity, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.rectangle(vid, (x1, y1), (x2, y2), (255, 0, 0), 2)
                s.add(identity)
            else:
                cv2.putText(vid, 'Unknown', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.rectangle(vid, (x1, y1), (x2, y2), (0, 255, 0), 2)

        back[174:174 + 500, 121:121 + 800] = vid
        cv2.imshow('Video Feed', back)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    show_attendance()


root = Tk()
selected_names = {}

def close():
    root.destroy()

def save():
    with open('attendance.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Name"])
        for name, var in selected_names.items():
            if var.get() == 1:  
                writer.writerow([name])
    print("Attendance saved.")

def show_attendance():
    root.geometry('400x500')
    root.title("Recognised Students")

    for i in s:
        #intVAr se checkbutton ki true or false value store aur update hoti hai
        var = IntVar()
        selected_names[i] = var
        checkbox = Checkbutton(root, text=i, variable=var, font=('Arial', 15, 'bold'), padx=15, pady=15)
        checkbox.pack(pady=2)

    save_button = Button(root, text="Save", command=save)
    save_button.pack(pady=10)

    close_button = Button(root, text="Close", command=close)
    close_button.pack(pady=10)

    root.mainloop()

video_capture()
