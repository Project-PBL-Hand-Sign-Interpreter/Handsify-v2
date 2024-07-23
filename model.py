import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed
from tensorflow.keras.callbacks import TensorBoard

DATA_PATH = 'CHANGE TO NEWEST DATASET'
KERAS_PATH = 'keras'
MODEL_BASE_NAME = 'test_'

NO_KEYPOINTS = 21*3 + 21*3

# actions = np.array(['Ada', 'Anda', 'Apa', 'Atau', 'Bantu', 
#                     'Banyak', 'Beli', 'Bisa', 'Dengan', 'Dingin',
#                     'Gula', 'Hallo', 'Ibu', 'Ini', 'Kakak', 'Kopi',
#                     'Malam', 'Pagi', 'Pak', 'Panas', 'Saya', 'Sedang',
#                     'Sedikit', 'Selamat', 'Siang', 'Terimakasih',
#                     'Tertarik', 'Untuk', 'Yang'])
actions = np.array(['Apa', 'Anda', 'Bisa'])
no_sequences = 4
no_frames = 20

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

def prepare_sequences(data_path, actions, no_sequences, no_frames):
    sequences, labels = [], []
    label_map = {label: num for num, label in enumerate(actions)}

    for action in actions:
        for sequence in range(no_sequences):
            window = []
            for frame in range(no_frames):
                res = np.load(os.path.join(data_path, action, str(sequence), "{}.npy".format(frame)))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])

    X = np.array(sequences)
    y = to_categorical(labels).astype(int)
    return X, y

def read_existing_models():
    no_models = 0
    exist = True

    while exist:
        if (os.path.exists(os.path.join(KERAS_PATH, MODEL_BASE_NAME + str(no_models) + ".h5"))):
            no_models += 1
        else:
            exist = False
    
    return no_models

model_number = read_existing_models()
model_name = MODEL_BASE_NAME + str(model_number)

X, y = prepare_sequences(DATA_PATH, actions, no_sequences, no_frames)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', name='lstm_1', input_shape=(no_frames, NO_KEYPOINTS)))
model.add(LSTM(128, return_sequences=True, activation='relu', name='lstm_2'))
model.add(LSTM(64, return_sequences=False, activation='relu', name='lstm_3'))
model.add(Dense(64, activation='relu', name='dense_1'))
model.add(Dense(32, activation='relu', name='dense_2'))
model.add(Dense(actions.shape[0], activation='softmax', name='output_dense'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.summary()

model.fit(X_train, y_train, epochs=2000, callbacks=[tb_callback])

model.save('keras/{}.h5'.format(model_name))