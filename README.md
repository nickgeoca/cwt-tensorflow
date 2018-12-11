# Tensorflow CWT
This implements a 1-D Continuous Wavelet Transform (CWT) in tensorflow. The benefit is that it runs parallel on GPUs.

The following wavelets are available:
* Ricker wavelet - cwtRicker
* Mortlet wavelet - cwtMortlet

## Benchmarks
Regarding CWT performance of Tensorflow vs Pywavelet, Pywavelet is about 13 times faster. However, this is a CPU only benchmark without using performance extensions, like AVX, on Tensorflow.

| Col1          | Col2          | Result  | Notes |
| ------------- |:-------------:| -----:| --- |
| Tensorflow CWT - GPU | Tensorflow CWT - CPU | GPU ~8x faster                | old laptop i5 vs GTX 750 TI ~1,400 GFLOPS | 
| Tensorflow CWT - CPU | Pywavelet CWT - CPU  | Pywavelet CWT ~13x faster     | Tensorflow w/o AVX extensions, etc |
| Tensorflow CWT - CPU | Pywavelet DWT - CPU  | Pywavelet DWT ~200,000 faster | Haar wavelet; Tensorflow w/o AVX extensions, etc |

### Benchmark times 
This can be aquired by running `python benchmark.py`

* DWT - sampleSize = 10000000
    * pywavelet dwt haar: 0.06824707984924316
    * pywavelet dwt db2: 0.08141493797302246
    * pywavelet dwt db8: 0.14669179916381836
* CWT - sampleSize = 10000; cwtWidth = 256
    * pywavelet cwt mortlet: 1.1284675598144531
    * tensorflow cwt mortlet: 14.783239364624023

## Examples
* [wavExample.py](https://github.com/nickgeoca/cwt-tensorflow/blob/master/wavExample.py). The audio sample rate is scaled down to 8000 samples per second (instead of typical 44100).

* [sinExample.py](https://github.com/nickgeoca/cwt-tensorflow/blob/master/sinExample.py). It produces the plot below. The wavelet used is shown below (scale=32).
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
