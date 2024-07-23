import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import tkinter.messagebox
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

def draw_styled_landmarks(image, results):

    # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
    #                          mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
    #                          mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
    #                          )

    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(0,22,76), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(0,44,250), thickness=2, circle_radius=2)
                             )

    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )

def extract_keypoints(results):
    # face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

def get_existing_data(start, no_sequences):
    action_index = 0
    sequence = start
    exist = True
    
    for i in range(action_index, actions.size):
        for j in range(sequence, sequence + no_sequences):
            if not os.path.exists(os.path.join(DATA_PATH, actions[i], str(j), "0.npy")):
                exist = False
                action_index = i
                sequence = j
                break
        if not exist:
            break

    return action_index, sequence

DATA_PATH = os.path.join('dataset_test_2')

# Rafi = 0
# Adji = 50
# Alfian = 100
# Benaya = 150
start = 0

# actions = np.array(['Ada', 'Anda', 'Apa', 'Atau', 'Bantu',
#                     'Banyak', 'Beli', 'Bisa', 'Dengan', 'Dingin',
#                     'Gula', 'Hallo', 'Ibu', 'Ini', 'Kakak', 'Kopi',
#                     'Malam', 'Pagi', 'Pak', 'Panas', 'Saya', 'Sedang',
#                     'Sedikit', 'Selamat', 'Siang', 'Terimakasih',
#                     'Tertarik', 'Untuk', 'Yang'])
actions = np.array(['Apa', 'Anda', 'Bisa'])

no_sequences = 50
no_frames = 20

for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence + start)))
        except:
            pass

cap = cv2.VideoCapture(0)
# Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
    action, sequence = get_existing_data(start, no_sequences)

    while cap.isOpened():
        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)

        # Draw landmarks
        draw_styled_landmarks(image, results)

        # Making a copy of the original image for changing texts
        original_image = image.copy()

        # Write action word
        cv2.putText(image, "Word : {}, Sequence : {}".format(actions[action], sequence), (15,12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        # Show image
        cv2.imshow("OpenCV Feed", image)

        if cv2.waitKey(5) & 0xFF == ord('s'):
            for no_frame in range(no_frames):
                image = original_image.copy()

                # NEW Apply wait logic
                if no_frame == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120,200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.waitKey(1000)

                cv2.putText(image, 'Collecting frames for {} Sequence Number {}'.format(actions[action], sequence), (15,12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                # Show to screen
                cv2.imshow('OpenCV Feed', image)

                # NEW Export keypoints
                keypoints = extract_keypoints(results)
                print(keypoints)
                npy_path = os.path.join(DATA_PATH, actions[action], str(sequence), str(no_frame))
                np.save(npy_path, keypoints)

            image = original_image.copy()
            cv2.putText(image, 'SAVED', (120, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
            cv2.imshow('OpenCV Feed', image)
            cv2.waitKey(1000)
            
            if action == actions.size - 1 and sequence == no_sequences - 1:
                image = original_image.copy()
                cv2.putText(image, 'FINISHED COLLECTING DATA', (120, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                cv2.imshow('OpenCV Feed', image)
                cv2.waitKey(1000)
                break

            sequence += 1

            if sequence - start == no_sequences:
                sequence = 0
                action += 1

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
                    
    cap.release()
    cv2.destroyAllWindows()
    for i in range (1,5):
        cv2.waitKey(1)