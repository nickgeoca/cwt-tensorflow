import tensorflow as tf
import numpy as np

def cwtRicker(wav, widthCwt):
    """Continuous Wavelet Transform using Ricker wavelet 
    Parameters
    ----------
    wav:      matrix - float32 - shape (N,)
    widthCwt: scalar - int32

    Returns
    -------
    output:   matrix - float32 - shape (widthCwt, N)
    """
    length = wav.shape[0]
    wav = tf.to_float(wav)
    wav = tf.reshape(wav, [1,length,1,1])

    # While loop functions
    def body(i, m): 
        v = conv1DWavelet(wav, i, rickerWavelet)
        v = tf.reshape(v, [length, 1])

        m = tf.concat(1, [m,v])

        return [1 + i, m]

    def cond_(i, m):
        return tf.less_equal(i, widthCwt)

    # Initialize and run while loop
    emptyCwtMatrix = tf.zeros([length, 0], dtype='float32') 
    i = tf.constant(1)
    _, result = tf.while_loop(cond_, body, [i, emptyCwtMatrix], back_prop=False, parallel_iterations=1024)
    result = tf.transpose(result)

    return result


def rickerWavelet(scale, sampleCount):
    scale = tf.to_float(scale)
    sampleCount = tf.to_float(sampleCount)

    def rickerWaveletEquationPart(time): 
        time = tf.to_float(time)
        
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
        
    unscaledTimes   = tf.to_float(tf.range(tf.to_int32(sampleCount))) - (sampleCount - 1.) / 2.
    times           = unscaledTimes / scale
    unscaledSamples = rickerWaveletEquationPart(times)
    samples         = unscaledSamples * scale ** -.5

    return samples

def conv1DWavelet(wav, waveletWidth, waveletEquation):
    kernelSamples = waveletWidth * 10
    kernel = waveletEquation(waveletWidth, kernelSamples)
    kernel = tf.reshape(kernel, tf.pack([kernelSamples,1,1,1]))

    conv = tf.nn.conv2d(wav, kernel, [1,1,1,1], padding='SAME') 
    conv = tf.squeeze(tf.squeeze(conv))

    return conv
