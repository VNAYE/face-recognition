import cv2
import numpy as np
import pickle
import streamlit as st

def load_model():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('TrainingImageLabel/trainner.yml')
    return recognizer

def recognize_faces_live(mark_attendance):
    recognizer = load_model()
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    names = {}
    
    with open('TrainingImageLabel/names.pkl', 'rb') as f:
        names = pickle.load(f)
    
    cam = cv2.VideoCapture(0)
    stframe = st.empty()

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            Id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
            if confidence < 50:
                name = names.get(Id, "Unknown")
                roll_number = Id
                mark_attendance(name, roll_number)
                cv2.putText(img, f"{name}-{Id}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            else:
                cv2.putText(img, "Your not registered please register", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Convert the image to RGB format for Streamlit
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        stframe.image(img_rgb, channels="RGB")

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

def recognize_face_from_image(mark_attendance, image_path):
    """
    Recognize a face from a single image file and mark attendance if recognized.
    """
    recognizer = load_model()
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    names = {}
    with open('TrainingImageLabel/names.pkl', 'rb') as f:
        names = pickle.load(f)
    img = cv2.imread(image_path)
    if img is None:
        st.error("Could not read the uploaded image.")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    found = False
    for (x, y, w, h) in faces:
        Id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        if confidence < 50:
            name = names.get(Id, "Unknown")
            roll_number = Id
            mark_attendance(name, roll_number)
            st.success(f"Attendance marked for {name} ({roll_number})")
            found = True
        else:
            st.warning("Face not recognized. Please register first.")
    if not found:
        st.warning("No recognizable face found in the image.")

if __name__ == "__main__":
    recognize_faces_live(lambda name, roll_number: print(f"Marked attendance for {name} ({roll_number})"))
