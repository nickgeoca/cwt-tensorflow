import tensorflow as tf
import matplotlib.pyplot as plt 
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wf
from cwt import cwtMortlet, cwtRicker

# Create 1-D wave
sampleRate,signal = wf.read('hendrixRiff.wav')
sampleCount = len(signal)
cwtWidth = 64

# Create tensorflow ops
cwtOp = cwtMortlet(tf.float32, signal, cwtWidth)

# Run tensorflow
sess = tf.Session()
cwt = sess.run(cwtOp)
sess.close()

# Plot signal, wavelet, cwt
f, axarr = plt.subplots(2, sharex=True)

axarr[0].plot(signal)               
axarr[0].set_title('Signal')

axarr[1].imshow(cwt, aspect='auto', interpolation='none') 
axarr[1].set_title('CWT')

f.subplots_adjust(hspace=0.3, left=.1, bottom=.05, top=.95, right=.95)
plt.show()
