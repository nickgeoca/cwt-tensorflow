import tensorflow as tf
import matplotlib.pyplot as plt 
import numpy as np
import scipy.signal as signal

from cwt import cwtRicker

widthCwt = 64
wav = np.sin(np.arange(1000) / 20.)

sess = tf.Session()
cwtOp = cwtRicker(wav, widthCwt)
result = sess.run(cwtOp)
sess.close()

result = result[1]
result = np.transpose(result)

plt.imshow(result, aspect='auto', interpolation='nearest') 
plt.show()






