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

#mtcnn ka use kiya h detectino k liye caffee was waste
detector = MTCNN()
MyFaceNet = FaceNet()

#dataset ka pth
folder = 'dataset1/'
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
            for file in listdir(f'C://Users//Dell//OpenCV//dataset1//{filename}'):
                vid = cv2.imread(f'C://Users//Dell//OpenCV//dataset1//{filename}//{file}')

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


cap = cv2.VideoCapture(0)
#ye hai vo set jisme identified logo ke name save honge
s = set()

def video_capture():
    #ye screen capture k liye
    _, vid = cap.read()
    vid = cv2.resize(vid, (800,600))
    vid = cv2.cvtColor(vid, cv2.COLOR_BGR2RGB)
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
        identity = 'Unknown'
        for key, value in database.items():
            #is version me humlog cosine similarity bhi check krenge, that will improve accuracy a lot
            similarity = cosine_similarity(value, signature)[0][0]
            if similarity > max_similarity:
                max_similarity = similarity
                identity = key

            if max_similarity > 0.6:  #threshold valu
                cv2.putText(vid, identity, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.rectangle(vid, (x1, y1), (x2, y2), (255, 0, 0), 2)
                s.add(identity)
            else:
                cv2.putText(vid, 'Unknown', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.rectangle(vid, (x1, y1), (x2, y2), (0, 255, 0), 2)

    #frame ko tkinter label pe display krne keliye convert krna pdega into image and then PIL image
    img = Image.fromarray(vid)        
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)
    
    label.after(1, video_capture)
        
#to write into csv
def save():
    with open('attendance.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Name"])
        for name in s:
            writer.writerow([name])

# exit keliye
def quit():
    save()
    root.destroy()

#initialixe tkinter window
root = Tk()
root.attributes("-fullscreen", True)
root.title("Video Feed")

label = Label(root)
#pack se window pe place ho jata h
label.pack()

btn = Button(root, text="Quit", command=quit)
btn.pack(pady=10)

video_capture()


root.mainloop()


if len(s) > 0:
    for i in s:
        print(i)
else:
    print("Empty")