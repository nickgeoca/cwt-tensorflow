TODO
 * add: samples = min(10*width, len(wav))
 * consier scipy's cwt ability of [1,1.5,2,2.5,3]. Currently does range(1,n)



# Useage: 

import tensorflow as tf
import matplotlib.pyplot as plt 
from scipy.io import wavfile
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

