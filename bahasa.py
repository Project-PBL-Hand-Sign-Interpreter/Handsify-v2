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
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed
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
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections

def draw_styled_landmarks(image, results):
   
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
   
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
     
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 
    
cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        print(results)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)

        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break 
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    for i in range (1,5):
        cv2.waitKey(1)

import os
import numpy as np

DATA_PATH = 'dataset'
actions = np.array(['Ada', 'Anda', 'Apakah', 'Atau', 'Bantu', 
                    'Banyak', 'Beli', 'Bisa', 'Dengan', 'Dingin',
                    'Gula', 'Hallo', 'Ibu', 'Ini', 'Kakak', 'Kopi',
                    'Malam', 'Pagi', 'Pak', 'Panas', 'Saya', 'Sedang',
                    'Sedikit', 'Selamat', 'Siang', 'Terimakasih',
                    'Tertarik', 'Untuk', 'Yang'])

no_sequences = 30

sequence_length = 30

def load_dataset(data_path):
    dataset = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.npy'):
                file_path = os.path.join(root, file)
                data = np.load(file_path)
                dataset.append(data)
    return dataset

dataset = load_dataset(DATA_PATH)
print(f"Loaded {len(dataset)} samples.")


def load_dataset_by_action(data_path, actions):
    dataset = {action: [] for action in actions}
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.npy'):
                file_path = os.path.join(root, file)
                data = np.load(file_path)
                # Assuming that the subfolder name indicates the action
                action = os.path.basename(os.path.dirname(root))
                if action in dataset:
                    dataset[action].append(data)
    return dataset

dataset_by_action = load_dataset_by_action(DATA_PATH, actions)
for action, data in dataset_by_action.items():
    print(f"Loaded {len(data)} samples for action '{action}'.")


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

result_test = extract_keypoints(results)

for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

label_map = {label:num for num, label in enumerate(actions)}

label_map

sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

np.array(sequences).shape

X = np.array(sequences)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)


model = Sequential()
model.add(TimeDistributed(Dense(units=128, activation='tanh'), input_shape=(30, 1662), name='time_distributed_dense'))
model.add(LSTM(128, return_sequences=True, activation='tanh', name='lstm_1'))
model.add(Dropout(0.2, name='dropout_1'))
model.add(LSTM(64, return_sequences=False, activation='tanh', name='lstm_2'))
model.add(Dropout(0.2, name='dropout_2'))
model.add(Dense(32, activation='relu', name='dense_1'))
model.add(Dropout(0.2, name='dropout_3'))
model.add(Dense(actions.shape[0], activation='softmax', name='output_dense'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.summary()


model.fit(X_train, y_train, epochs=1000, callbacks=[tb_callback])

res = model.predict(X_test)
actions[np.argmax(res[5])]
actions[np.argmax(y_test[5])]

model.save('action.h5')

from tensorflow import keras
model = keras.models.load_model('action.h5')

model.load_weights('action.h5')

from tkinter import *
import tkinter.messagebox
import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe holistic model
mp_holistic = mp.solutions.holistic
actions = np.array(['Ada', 'Anda', 'Apakah', 'Atau', 'Bantu', 
                    'Banyak', 'Beli', 'Bisa', 'Dengan', 'Dingin',
                    'Gula', 'Hallo', 'Ibu', 'Ini', 'Kakak', 'Kopi',
                    'Malam', 'Pagi', 'Pak', 'Panas', 'Saya', 'Sedang',
                    'Sedikit', 'Selamat', 'Siang', 'Terimakasih',
                    'Tertarik', 'Untuk', 'Yang'])
# Assuming mediapipe_detection, draw_styled_landmarks, extract_keypoints, and model are defined elsewhere
# Assuming actions is defined elsewhere

root = Tk()
root.geometry('500x300')
root.resizable(width=True, height=True)
frame = Frame(root, relief=RIDGE, borderwidth=2)
frame.pack(fill=BOTH, expand=1)
root.title('Bahasa Isyarat Indonesia')
frame.config(background='lightgreen')
label = Label(frame, text="\nDeteksi Bahasa Isyarat Indonesia \n", bg='lightgreen', font=('Quicksand 14 bold'))
label.pack(side=TOP)

sentences = []

def web():
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.5

    cap = cv2.VideoCapture(0)
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            # Read feed
            ret, frame = cap.read()

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            print(results)
                    
            # Draw landmarks
            draw_styled_landmarks(image, results)
        
            # Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]
        
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                response = actions[np.argmax(res)]
                print(response)
                
                if(len(sentences) == 0):
                    sentences.append(response)
                elif(sentences[-1] != response):
                    sentences.append(response)
                else:
                    pass
                
                predictions.append(np.argmax(res))
            
                # Viz logic
                if np.unique(predictions[-10:])[0] == np.argmax(res): 
                    if res[np.argmax(res)] > threshold: 
                        if len(sentence) > 0: 
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                    if len(sentence) > 5: 
                        sentence = sentence[-5:]

                cv2.putText(image, ' '.join(sentence), (3, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3, cv2.LINE_AA)
        
            # Show to screen
            cv2.imshow('Detection', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        for i in range(1, 5):
            cv2.waitKey(1)

but1 = Button(frame, padx=5, pady=5, width=20, bg='white', fg='black', relief=GROOVE, command=web, text='START DETECTION', font=('helvetica 15 bold'))
but1.place(x=125, y=154)
root.mainloop()
print(sentences)

unique_words = list(set(sentences))
print(unique_words)
print(len(unique_words))