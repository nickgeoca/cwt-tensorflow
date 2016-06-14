# Tensorflow CWT
This is implements a 1-D Continuous Wavelet Transform (CWT) using a Ricker wavelet in tensorflow. It is very similar to scipy's cwt routine, excpet is slightly limited but is much faster.

It has the advantage of running in parallel on a GPU and is about 8x faster than an old laptop i5 using a GTX 750 TI (~1.3GPLOPS). This is done by using tensorflow's while_loop.


## TODO
* Add this in code similar to scipy's [cwt](https://github.com/scipy/scipy/blob/63bcdc4eeafa59553c00e44343dbb38380bd9d45/scipy/signal/wavelets.py#L362): samples = min(10*width, len(wav))
* consier scipy's ability to specify the wavelet scale
```python
# Scipy's cwt can specify the wavelet scales in detail. This api can't do that.
cwt(wav, signal.ricker, [1,1.5,2,2.5,3])
# This api is equivilent to calling scipy's cwt as below.
cwt(wav, signal.ricker, range(1,n))
```


## Usage
```python
import tensorflow as tf
import matplotlib.pyplot as plt 
import numpy as np
from cwt import cwtRicker

widthCwt = 64
wav = np.sin(np.arange(1000) / 20.)

sess = tf.Session()
cwtOp = cwtRicker(wav, widthCwt)
result = sess.run(cwtOp)
sess.close()

result = result[1]
result = np.transpose(result)
```
