import tensorflow as tf
import numpy as np

# Run the cwtRicker, cwtMortlet

"""
Continuous Wavelet Transforms
    Parameters
    ----------
    wav:      matrix - float32 - shape (N,)
    widthCwt: scalar - int32

    Returns
    -------
    output:   matrix - float32 - shape (widthCwt, N)
"""
def cwtRicker(tfType, wav, widthCwt):  return cwt(tfType, wav, widthCwt, rickerWavelet)
def cwtMortlet(tfType, wav, widthCwt): return cwt(tfType, wav, widthCwt, mortletWavelet)

# ------------------------------------------------------

def cwt(tfType, wav, widthCwt, wavelet):
    length = wav.shape[0]
    wav = tf.cast(wav, tfType)
    wav = tf.reshape(wav, [1,length,1,1])

    # While loop functions
    def body(i, m): 
        v = conv1DWavelet(tfType, wav, i, wavelet)
        v = tf.reshape(v, [length, 1])

        m = tf.concat([m,v], 1)

        return [1 + i, m]

    def cond_(i, m):
        return tf.less_equal(i, widthCwt)

    # Initialize and run while loop
    emptyCwtMatrix = tf.zeros([length, 0], tfType) 
    i = tf.constant(1)
    _, result = tf.while_loop(
            cond_,
            body,
            [i, emptyCwtMatrix],
            shape_invariants=[i.get_shape(), tf.TensorShape([length, None])],
            back_prop=False,
            parallel_iterations=1024,
            )
    result = tf.transpose(result)

    return result

# ------------------------------------------------------
#                 wavelets
def rickerWavelet(tfType, scale, sampleCount):
    def waveEquation(time): 
        time = cast(time, tfType)
        
        tSquare = time ** 2.
        sigma   = 1.
        sSquare = sigma ** 2.

        # _1 = 2 / ((3 * a) ** .5 * np.pi ** .25)
        _1a = (3. * sigma) ** .5
        _1b = np.pi ** .25
        _1 = 2. / (_1a * _1b)

        # _2 = 1 - t**2 / a**2
        _2 = 1. - tSquare / sSquare

        # _3 = np.exp(-(t**2) / (2 * a ** 2))
        _3a = -1. * tSquare
        _3b = 2. * sSquare
        _3 = tf.exp(_3a / _3b)

        return _1 * _2 * _3
        
    return waveletHelper(tfType, scale, sampleCount, waveEquation)

def mortletWavelet(tfType, scale, sampleCount):
    def waveEquation(time): 
        return tf.exp(-1. * time ** 2. / 2.) * tf.cos(5. * time) # https://www.mathworks.com/help/wavelet/ref/morlet.html

    return waveletHelper(tfType, scale, sampleCount, waveEquation)

def waveletHelper(tfType, scale, sampleCount, waveEquation):
    scale         = tf.cast(scale, tfType)
    sampleCount   = tf.cast(sampleCount, tfType)
    unscaledTimes = tf.cast(tf.range(tf.to_int64(sampleCount)), tfType) - (sampleCount - 1.) / 2.
    times         = unscaledTimes / scale
    wav           = waveEquation(times)
    wav           = wav * scale ** -.5
    return wav

# ------------------------------------------------------
#                    helpers
def conv1DWavelet(tfType, wav, waveletWidth, waveletEquation):
    kernelSamples = waveletWidth * 10
    kernel = waveletEquation(tfType, waveletWidth, kernelSamples)
    kernel = tf.reverse(kernel, [0])
    kernel = tf.reshape(kernel, tf.stack([kernelSamples,1,1,1]))

    conv = tf.nn.convolution(wav, kernel, 'SAME')
    conv = tf.squeeze(tf.squeeze(conv))

    return conv

