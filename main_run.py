import cv2
import numpy as np
import mediapipe as mp
from tkinter import *
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import urllib


mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

actions = np.array(['Ada', 'Anda', 'Apa', 'Atau', 'Bantu',
                    'Banyak', 'Beli', 'Bisa', 'Dengan', 'Dingin',
                    'Gula', 'Hallo', 'Ibu', 'Ini', 'Kakak', 'Kopi',
                    'Malam', 'Pagi', 'Pak', 'Panas', 'Saya', 'Sedang',
                    'Sedikit', 'Selamat', 'Siang', 'Terimakasih',
                    'Ingin', 'Untuk', 'Yang'])
# actions = np.array(["Apa", "Anda", "Bisa"])

model = keras.models.load_model('keras/test_4.h5')


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_styled_landmarks(image, results):

    # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
    #                          mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
    #                          mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
    #                          )

    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(
                                  color=(121, 44, 250), thickness=2, circle_radius=2)
                              )
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(
                                  color=(245, 66, 230), thickness=2, circle_radius=2)
                              )


def extract_keypoints(results):
    # face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten(
    # ) if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten(
    ) if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten(
    ) if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])

def is_showing_hand(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten(
    ) if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten(
    ) if results.right_hand_landmarks else np.zeros(21*3)

    return np.any(lh) or np.any(rh)

def web():
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.7


    # cap = cv2.VideoCapture(0);
    url = "http://192.168.220.59/cam-hi.jpg"
    # Set mediapipe model
    with mp_holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
        while True:
        # while cap.isOpened() :
            # Read feed
            imgResp = urllib.request.urlopen(url)
            imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)
            img = cv2.imdecode(imgNp, -1)

            # ret, frame = cap.read()

            # Make detections
            image, results = mediapipe_detection(img, holistic)

            # image, results = mediapipe_detection(frame, holistic);

            # Draw landmarks
            draw_styled_landmarks(image, results)

            # Prediction logic
            if is_showing_hand(results):
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-20:]
            else:
                sequence = []

            if len(sequence) == 20:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                response = actions[np.argmax(res)]
                print(response)

                if (len(sentences) == 0):
                    sentences.append(response)
                elif (sentences[-1] != response):
                    sentences.append(response)
                else:
                    pass

                if len(sentence) > 0:
                    if response == sentence[-1]:
                        sentence.append(" . " + response)
                else:
                    sentence.append(response)

                if len(sentence) > 5:
                    sentence = sentence[-5:]

                predictions.append(np.argmax(res))

                #Viz logic
                if np.unique(predictions[-10:])[0] == np.argmax(res):
                    if res[np.argmax(res)] > threshold:
                        if len(sentence) > 0 and response != sentence[-1]:
                                sentence.append(response)
                        else:
                            sentence.append(response)

                        if len(sentence) > 10:
                            sentence = sentence[-5:]
                        
                        sequence = []

            cv2.putText(image, ' '.join(sentence), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3, cv2.LINE_AA)
            
            # Show to screen
            cv2.imshow('Detection', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        for i in range (1, 5) :
            cv2.waitKey(1)



root = Tk()
root.geometry('500x300')
root.resizable(width=True, height=True)
frame = Frame(root, relief=RIDGE, borderwidth=2)
frame.pack(fill=BOTH, expand=1)
root.title('Bahasa Isyarat Indonesia')
frame.config(background='lightgreen')
label = Label(frame, text="\nDeteksi Bahasa Isyarat Indonesia \n",
              bg='lightgreen', font=('Quicksand 14 bold'))
label.pack(side=TOP)

sentences = []

but1 = Button(frame, padx=5, pady=5, width=20, bg='white', fg='black',
              relief=GROOVE, command=web, text='START DETECTION', font=('helvetica 15 bold'))
but1.place(x=125, y=154)
root.mainloop()
print(sentences)

unique_words = list(set(sentences))
print(unique_words)
print(len(unique_words))