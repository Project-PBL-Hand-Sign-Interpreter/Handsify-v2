import os
import numpy as np

actions = np.array(['Ada', 'Anda', 'Apa', 'Atau', 'Bantu',
                    'Banyak', 'Beli', 'Bisa', 'Dengan', 'Dingin',
                    'Gula', 'Hallo', 'Ibu', 'Ini', 'Kakak', 'Kopi',
                    'Malam', 'Pagi', 'Pak', 'Panas', 'Saya', 'Sedang',
                    'Sedikit', 'Selamat', 'Siang', 'Terimakasih',
                    'Tertarik', 'Untuk', 'Yang'])

DATA_PATH = "dataset_test_3"

start = 50
no_sequences = 150

for action in actions:
    for i in range(start, start + no_sequences):
        old_folder = os.path.join(DATA_PATH, action, str(i))
        new_folder = os.path.join(DATA_PATH, action, str(i - start))
        os.rename(old_folder, new_folder)