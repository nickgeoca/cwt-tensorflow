import tensorflow as tf
import matplotlib.pyplot as plt 
import numpy as np
import scipy.signal as signal
from cwt import cwtMortlet, cwtRicker, mortletWavelet, rickerWavelet

# Create 1-D wave
sampleSize = 1000
cwtWidth = 256
signal = np.sin(np.arange(sampleSize) / 20.)

# Create tensorflow operations
cwtOp = cwtMortlet(tf.float32, signal, cwtWidth)
waveletOp = mortletWavelet(tf.float32, 32, sampleSize) # Scale value 32. We are using from 1 to 256 (cwtWidth parameter). After scale of 160 it gets less accurate with sampleSize of 1000

# Run tensorflow
sess = tf.Session()
cwt = sess.run(cwtOp)
wavelet = sess.run(waveletOp)
sess.close()

# Plot signal, wavelet, cwt
f, axarr = plt.subplots(3, sharex=True)
axarr[0].plot(signal)   
axarr[0].set_title('Signal')

axarr[1].plot(wavelet)   
axarr[1].set_title('Wavelet')

axarr[2].imshow(cwt, aspect='auto', interpolation='nearest') 
axarr[2].set_title('CWT')

f.subplots_adjust(hspace=0.3, left=.1, bottom=.05, top=.95, right=.95)
plt.show()
