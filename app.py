import streamlit as st
import pandas as pd
import os
import cv2
import numpy as np
import face_recognition
import pickle
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# === Initialization ===

if not os.path.exists("faces"):
    os.makedirs("faces")
if not os.path.exists("encodings"):
    os.makedirs("encodings")
ENCODINGS_PATH = "encodings/encodings.pkl"

# === Helper Functions ===

def save_student_details(name, roll_number):
    if not os.path.exists('students.csv'):
        df = pd.DataFrame(columns=['Name', 'RollNumber'])
    else:
        df = pd.read_csv('students.csv')
    
    if not ((df['Name'] == name) & (df['RollNumber'] == roll_number)).any():
        new_entry = pd.DataFrame({'Name': [name], 'RollNumber': [roll_number]})
        df = pd.concat([df, new_entry], ignore_index=True)
        df.drop_duplicates(subset=['Name', 'RollNumber'], keep='first', inplace=True)
        df.to_csv('students.csv', index=False)
        st.success(f"Student {name} registered successfully.")
    else:
        st.warning("Student already registered")

def mark_attendance(name, roll_number):
    date = datetime.now().strftime('%Y-%m-%d')
    if not os.path.exists('attendance.csv'):
        df = pd.DataFrame(columns=['Name', 'RollNumber', 'Date', 'Present'])
    else:
        df = pd.read_csv('attendance.csv')
    
    if not ((df['Name'] == name) & (df['RollNumber'] == roll_number) & (df['Date'] == date)).any():
        new_entry = pd.DataFrame({'Name': [name], 'RollNumber': [roll_number], 'Date': [date], 'Present': ['Yes']})
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv('attendance.csv', index=False)
        st.success(f"Attendance marked for {name}")
    else:
        st.info(f"Attendance already marked today for {name}")

def capture_and_encode_face(name, roll_number):
    cap = cv2.VideoCapture(0)
    st.info("Capturing face. Press 'q' to capture when ready.")

    face_image_path = f"faces/{name}-{roll_number}.jpg"

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam.")
            break
        cv2.imshow("Press 'q' to capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite(face_image_path, frame)
            break

    cap.release()
    cv2.destroyAllWindows()

    # Extract face encoding
    image = face_recognition.load_image_file(face_image_path)
    face_locations = face_recognition.face_locations(image)
    if not face_locations:
        st.error("No face detected in the image.")
        os.remove(face_image_path)
        return

    encoding = face_recognition.face_encodings(image, known_face_locations=face_locations)[0]

    # Load existing encodings
    encodings = []
    if os.path.exists(ENCODINGS_PATH):
        with open(ENCODINGS_PATH, "rb") as f:
            encodings = pickle.load(f)
    
    encodings.append({
        "name": name,
        "roll_number": roll_number,
        "encoding": encoding
    })

    # Save updated encodings
    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump(encodings, f)

    st.success("Face registered and encoded successfully.")

def load_known_faces():
    if not os.path.exists(ENCODINGS_PATH):
        return [], [], []
    with open(ENCODINGS_PATH, "rb") as f:
        encodings = pickle.load(f)
    names = [e["name"] for e in encodings]
    roll_numbers = [e["roll_number"] for e in encodings]
    known_encodings = [e["encoding"] for e in encodings]
    return known_encodings, names, roll_numbers

# === Video Processor for Real-Time Recognition ===

class AttendanceVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.known_encodings, self.names, self.roll_numbers = load_known_faces()
        self.attendance_marked = set()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_img)
        face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(self.known_encodings, face_encoding, tolerance=0.45)
            face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)

            if any(matches):
                best_match_index = np.argmin(face_distances)
                name = self.names[best_match_index]
                roll = self.roll_numbers[best_match_index]

                if (name, roll) not in self.attendance_marked:
                    mark_attendance(name, roll)
                    self.attendance_marked.add((name, roll))

                top, right, bottom, left = face_location
                cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(img, f"{name}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return frame.from_ndarray(img, format="bgr24")

# === Streamlit Interface ===

st.title("📸 Attendance Management System")

menu = ["Register Student", "Attendance", "View Databases"]
choice = st.sidebar.selectbox("📋 Menu", menu)

if choice == "Register Student":
    st.subheader("📝 Register Student with Face")
    name = st.text_input("Student Name")
    roll_number = st.text_input("Roll Number")

    if st.button("Register"):
        if name and roll_number:
            save_student_details(name, roll_number)
            capture_and_encode_face(name, roll_number)
        else:
            st.error("Please enter both name and roll number.")

elif choice == "Attendance":
    st.subheader("🎥 Live Face Recognition")
    webrtc_ctx = webrtc_streamer(
        key="attendance",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=AttendanceVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if webrtc_ctx.state.playing:
        st.info("Webcam active. Stand in front of camera to mark attendance.")
    else:
        st.warning("Click 'Start' to activate webcam.")

elif choice == "View Databases":
    st.subheader("📁 Registered Students")
    if os.path.exists('students.csv'):
        df = pd.read_csv('students.csv')
        st.dataframe(df)
    else:
        st.info("No students registered.")

    st.subheader("🕒 Attendance Records")
    if os.path.exists('attendance.csv'):
        df = pd.read_csv('attendance.csv')
        st.dataframe(df)
    else:
        st.info("No attendance records yet.")
