import pywt
import time
import numpy as np
from cwt import cwtMortlet, cwtRicker, mortletWavelet, rickerWavelet
import tensorflow as tf

sampleSize = 10000
cwtWidth = 256
signal = np.sin(np.arange(sampleSize) / 20.)

start = time.time()
a = pywt.dwt(signal, 'haar')
end = time.time()
print('pywt dwt haar: ' + str(end - start))

start = time.time()
a = pywt.dwt(signal, 'db2')
end = time.time()
print('pywt dwt db2: ' + str(end - start))

start = time.time()
a = pywt.dwt(signal, 'db8')
end = time.time()
print('pywt dwt db8: ' + str(end - start))

start = time.time()
coef, freqs=pywt.cwt(signal,np.arange(1,1+cwtWidth),'morl')
end = time.time()
print('pywt cwt mortlet: ' + str(end - start))

cwtOp = cwtMortlet(tf.float32, signal, cwtWidth)
sess = tf.Session()
start = time.time()
cwt = sess.run(cwtOp)
end = time.time()
sess.close()
print('tf cwt mortlet: ' + str(end - start))

