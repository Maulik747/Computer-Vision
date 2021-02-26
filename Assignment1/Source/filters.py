import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
    padded_img = np.zeros((Hi+Hk-1, Wi+Wk-1))
    padded_img[:-Hk//2,:-Wk//2]=image
    for m in range(Hk, Hi):
        for n in range(Wk, Wi):
            for k in range(Hk):
                for l in range(Wk):
                    out[m - Hk//2,n - Wk//2]+=padded_img[m-k,n-l]*kernel[k,l]
            
    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out=np.zeros((H+2*pad_height, W+2*pad_width))
    out[pad_height:H+pad_height, pad_width: W+pad_width]=image
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
    h = kernel
    h1 = np.flipud(h)
    h2 = np.fliplr(h1)
    Hk, Wk = np.shape(kernel) 
    Hi, Wi = np.shape(image)
    y = Hk//2
    x = Wk//2
    f = image
    f = zero_pad(f, y, x)
    out = np.copy(image)
    for i in range(y, Hi + y):
        for j in range(x, Wi + x):
            sum1 = f[i-y:i+y+1,j-x:j+x+1]
            out[i-y,j-x] = (sum1 * h2).sum()
    ### END YOUR CODE

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    Hf,Wf=f.shape
    out=np.zeros((Hf,Wf))
    g = np.flip(np.flip(g,axis=0), axis=1)
    out=conv_fast(f,g[0:-1])  #removing the end row from g

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    avg=(g.mean(axis=1)).mean(axis=0)
    c=g-avg
    out= cross_correlation(f,c)
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    favg=(f.mean(axis=1)).mean(axis=0)
    gavg=(g.mean(axis=1)).mean(axis=0)
    fstd=np.std(f)
    gstd=np.std(g)
    normf=(f-favg)/fstd
    normg=(g-gavg)/gstd
    out=cross_correlation(normf,normg)
    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out
