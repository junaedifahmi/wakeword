import numpy as np
from python_speech_features import mfcc
import scipy.io.wavfile as wav
from keras.models import load_model
import sys


def pad_along_axis(array: np.ndarray, target_length, axis=0):

    pad_size = target_length - array.shape[axis]
    axis_nb = len(array.shape)

    if pad_size < 0:
        return array

    npad = [(0, 0) for x in range(axis_nb)]
    npad[axis] = (0, pad_size)

    b = np.pad(array, pad_width=npad, mode='constant', constant_values=0)

    return b

def load_h5model(path):
    return load_model(path)
    
path = sys.argv[2]
model = sys.argv[1]
    
model = load_h5model(model)

sr, frame = wav.read(path)
x = mfcc(frame, sr)

if x.shape != (846, 13):
    # Padding
    if x.shape[0] < 846:
        x = pad_along_axis(x, 846,0)
    elif x.shape[0] > 846:
        x = x[:846,:]

x = np.reshape( x, (1,846,13) )
pred = model.predict(x)
idx = np.argmax(pred)
kelas = "Wakeword" if idx == 0 else "Not wake word"
hasil = {
    'prob' : pred[0][idx],
    'label' : kelas
}
print(hasil)