import tflite_runtime.interpreter as tflite
import sys
import numpy as np
from utils import read, MFCC

mfcc = MFCC()


def pad_along_axis(array: np.ndarray, target_length, axis=0):

    pad_size = target_length - array.shape[axis]
    axis_nb = len(array.shape)

    if pad_size < 0:
        return array

    npad = [(0, 0) for x in range(axis_nb)]
    npad[axis] = (0, pad_size)

    b = np.pad(array, pad_width=npad, mode='constant', constant_values=0)

    return b


interpreter = tflite.Interpreter(sys.argv[1])
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

sr, frames = read(sys.argv[2])

feat = mfcc.sig2s2mfc(frames)

if feat.shape != (846, 13):
    # Padding
    if feat.shape[0] < 846:
        feat = pad_along_axis(feat, 846, 0)
    elif feat.shape[0] > 846:
        feat = feat[:846,:]

feat = np.reshape(feat, input_details[0]['shape'])
feat = feat.astype(input_details[0]['dtype'])
interpreter.set_tensor(input_details[0]['index'], feat)

interpreter.invoke()

predicted = interpreter.get_tensor(output_details[0]['index'])
idx = np.argmax(predicted)
kelas = "wakeword" if idx == 0 else "not wakeword"
hasil = {
    'prob' : predicted[0][idx],
    'label' : kelas
}
print(hasil)

