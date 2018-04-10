# Tensorflow CWT
This implements a 1-D Continuous Wavelet Transform (CWT) using a Ricker wavelet in tensorflow. It is very similar to scipy's cwt routine.

It has the advantage of running in parallel on a GPU and is about 8x faster than an old laptop i5 using a GTX 750 TI (~1,400 GFLOPS). This is done by using tensorflow's parallel while_loop function.

The following wavelets are available:
* Ricker wavelet - cwtRicker
* Mortlet wavelet - cwtMortlet

## Examples
* [wavExample.py](https://github.com/nickgeoca/cwt-tensorflow/blob/master/wavExample.py). The audio sample rate is scaled down to 8000 samples per second (instead of typical 44100).

* [sinExample.py](https://github.com/nickgeoca/cwt-tensorflow/blob/master/sinExample.py). It produces the plot below. The wavelet used is show below (scale=32).
![](https://github.com/nickgeoca/cwt-tensorflow/blob/master/mortletCWT.png)

## Notes
* The wavelet can be undersampled if the scale is too small. An example of this is seen below- the scale was set to 1. 
<img src="https://github.com/nickgeoca/cwt-tensorflow/blob/master/undersampled-wavelet-p1.png" width="30%" height="30%"><img src="https://github.com/nickgeoca/cwt-tensorflow/blob/master/undersampled-wavelet-p2.png" width="30%" height="30%">

## Dev Notes
* This cwt and scipy's cwt both limit the Ricker wavelet samples to 10x the scale size to improve accuracy. 

## TODO
* Add this line of code similar to scipy's [cwt](https://github.com/scipy/scipy/blob/63bcdc4eeafa59553c00e44343dbb38380bd9d45/scipy/signal/wavelets.py#L362): samples = min(10*width, len(wav))
* consier scipy's ability to specify the wavelet scale
```python
# Scipy's cwt can specify the wavelet scales in detail. This api can't do that.
cwt(wav, signal.ricker, [1,1.5,2,2.5,3])
# This api is equivilent to calling scipy's cwt as below.
cwt(wav, signal.ricker, range(1,n))
```
* Maybe add 2d verison
