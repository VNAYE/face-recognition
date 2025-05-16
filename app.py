import streamlit as st
import pandas as pd
import os
from takeimages import capture_images
from train import train_model
from test import recognize_faces_live
from datetime import datetime

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


st.title("Attendance Management System")

menu = ["Register Student", "Attendance", "View Databases"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Register Student":
    st.subheader("Register Student")
    name = st.text_input("Enter the name of the student:", value="", max_chars=50, key="name", help="This field is required.")
    roll_number = st.text_input("Enter the roll number of the student:", value="", max_chars=20, key="roll_number", help="This field is required.")
    camera_image = st.camera_input("Capture student image")
    if st.button("Register"):
        if name and roll_number and camera_image is not None:
            save_student_details(name, roll_number)
            # Save the captured image to disk for training
            img_path = f"TrainingImage/{name}-{roll_number}-webcam.jpg"
            with open(img_path, "wb") as f:
                f.write(camera_image.getbuffer())
            # Optionally, you can modify capture_images to accept this image or skip it
            train_model()
            st.success(f"Student {name} registered")
        else:
            st.error("Please fill in all required fields and capture an image.")

elif choice == "Attendance":
    st.subheader("Attendance")
    camera_image = st.camera_input("Capture your image for attendance")
    if st.button("Start Attendance"):
        if camera_image is not None:
            # Save the captured image temporarily for recognition
            img_path = "temp_attendance.jpg"
            with open(img_path, "wb") as f:
                f.write(camera_image.getbuffer())
            # Use the new function for image-based recognition
            from test import recognize_face_from_image
            recognize_face_from_image(mark_attendance, img_path)
        else:
            st.error("Please capture your image for attendance.")

elif choice == "View Databases":
    st.subheader("Student Database")
    if os.path.exists('students.csv'):
        students_df = pd.read_csv('students.csv')
        students_df.drop_duplicates(subset=['Name', 'RollNumber'], keep='first', inplace=True)
        st.dataframe(students_df)
    else:
        st.write("No student data available.")
    
    st.subheader("Attendance Database")
    if os.path.exists('attendance.csv'):
        attendance_df = pd.read_csv('attendance.csv')
        st.dataframe(attendance_df)
    else:
        st.write("No attendance data available.")
