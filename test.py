import tensorflow as tf
import matplotlib.pyplot as plt 
import numpy as np
import scipy.signal as signal
from cwt import cwtRicker

# Create 1-D wave
widthCwt = 256
wav = np.sin(np.arange(1000) / 20.)

# Run CWT
sess = tf.Session()
cwtOp = cwtRicker(wav, widthCwt)
result = sess.run(cwtOp)
sess.close()

# Plot cwt and wave
plt.figure(1)
plt.subplot(211) # Plot wave
plt.plot(wav)   
plt.subplot(212) # Plot CWT of wave
plt.imshow(result, aspect='auto', interpolation='nearest') 
plt.show()






