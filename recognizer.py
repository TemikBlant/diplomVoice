from tensorflow.keras.models import load_model
from preprocess import preprocess_dataset
import os
import numpy as np
import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import Counter


class Recognizer:
    def __init__(self):
        temp_dir = pathlib.Path('temp')
        self.temp_file = temp_dir / 'test.wav'
        samples_dir = pathlib.Path('samples')
        self.numbers = np.array(tf.io.gfile.listdir(str(samples_dir)))

    def load_models(self):
        print('___________________________________start loading model______________________________')
        self.model1 = load_model('nets\\recognizer1.h5')
        self.model2 = load_model('nets\\recognizer2.h5')
        self.model3 = load_model('nets\\recognizer3.h5')
        print('___________________________________model has been loaded____________________________')

    def counter_for_models(self, answers):
        counter = Counter(answers).most_common(1)
        print(counter)
        if counter[0][1] >= 2:
            return counter[0][0]
        else:
            return 'Ой, не поняо=л чота'

    def recognize(self):
        sample_ds = preprocess_dataset([str(self.temp_file)])
        for spectrogram, label in sample_ds.batch(1):
            prediction1 = self.model1(spectrogram)
            prediction2 = self.model2(spectrogram)
            prediction3 = self.model3(spectrogram)
            answer1 = list(prediction1[0]).index(max(prediction1[0]))
            answer2 = list(prediction2[0]).index(max(prediction2[0]))
            answer3 = list(prediction3[0]).index(max(prediction3[0]))
            return self.counter_for_models([self.numbers[answer1], self.numbers[answer2], self.numbers[answer3]])
