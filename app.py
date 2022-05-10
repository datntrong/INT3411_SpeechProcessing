import sounddevice as sd
from scipy.io.wavfile import write
import soundfile as sf
import librosa
import numpy as np

# loadmodels
import pickle
class_names = ['xuong', 'len', 'phai', 'trai', 'nhay', 'ban', 'a', 'b']
model = {}
for key in class_names:
    name = f"models_train/model_{key}.pkl"
    with open(name, 'rb') as file:
        model[key] = pickle.load(file)


def get_mfcc(file_path):
    y, sr = librosa.load(file_path) # read .wav file

    hop_length = 256

    win_length = 512
    # mfcc is 13 x T matrix
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=13, n_fft=1024,
        hop_length=hop_length, win_length=win_length)
    # substract mean from mfcc --> normalize mfcc

    mfcc = np.subtract(mfcc,np.mean(mfcc))
    # delta feature 1st order and 2nd order
    delta1 = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    # X is 39 x T
    X = np.concatenate([mfcc, delta1, delta2], axis=0) # O^r
    # return T x 39 (transpose of X)
    return X.T # hmmlearn use T x N matrix

def record():
    filename = "audio.wav"

    # Sampling frequency
    freq = 22050

    # Recording duration
    duration = 1

    # Start recorder with the given values
    # of duration and sample frequency
    recording = sd.rec(int(duration * freq),
                       samplerate=freq, channels=1)

    # Record audio for the given number of seconds
    sd.wait()

    # This will convert the NumPy array to an audio
    # file with the given sampling frequency
    write(filename, freq, recording)


def play(filepath):
    data, fs = sf.read(filepath, dtype='float32')
    sd.play(data, fs)
    status = sd.wait()


def predict(filepath):

    #Predict
    record_mfcc = get_mfcc(filepath)

    scores = [model[cname].score(record_mfcc) for cname in class_names]
    # print(scores)
    predict_word = np.argmax(scores)
    print('Ket qua cua tu vua noi la : ',end='')
    print(class_names[predict_word])

if __name__ == '__main__':
    print(
        'Nhap 1 de record\nNhap 2 de play am thanh vua record\nNhap 0 de thoat chuong trinh\nNhap 3 de predict doan am thanh vua ghi\n'
    )
    while True:

        n = input()

        try:
            n = int(n)
            if n == 1:
                print('Start recording. Please say')
                record()
                print('Record done')
            if n == 2:
                filename = "audio.wav"
                play(filename)
            if n == 3:
                filename = "audio.wav"
                predict(filename)
            if n == 0:
                break
        except:
            print('',end='')
