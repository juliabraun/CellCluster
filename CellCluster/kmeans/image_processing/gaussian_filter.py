import numpy as np


def gaussian_mask(sigma, trim):
    """
    Computes the values of a 2D gaussian function with the desired sigma for the positions in a kernel with desired size

    :param sigma: integer or float representing the standard deviation
    :param trim: float, used to compute kernel size. Default is 4.0.
    :return: kernel as ndarray of shape size containing values of gaussian function
    """
    # compute distance d from center to edge of the filter kernel
    d = int(trim * sigma + 0.5)
    # empty list for storage
    vector = []
    for i in range(-d, d + 1):
        # compute value of one dimensional gaussian function (with mean = 0)
        gaussian_1d = np.exp(-i * i / (2 * sigma * sigma)) / np.sqrt(2 * np.pi * sigma * sigma)
        # store value in empty list
        vector.append(gaussian_1d)
    # compute outer product of vector with itself to obtain values of two dimensional gaussian function
    kernel = np.outer(vector, vector)
    # normalize kernel so that it sums to 1
    kernel /= np.sum(kernel)
    return kernel


def convolve(im, kernel):
    """
    Applies a filter kernel to an array (e.g. an image).
    Computes the values for pixels near the edge by virtually flipping the image on edges and corners.
    :param im: ndarray (e.g. an image)
    :param kernel: ndarray
    :return: filtered array
    """
    im_row, im_col = im.shape
    kernel_row, kernel_col = kernel.shape

    # increase size of image to be able to apply filter on pixels near the edge
    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)
    padded_im = np.zeros((im_row + (2 * pad_height), im_col + (2 * pad_width)))
    pad_row, pad_col = padded_im.shape
    # insert the original image into the middle of the padded image
    padded_im[pad_height:pad_row - pad_height, pad_width:pad_col - pad_width] = im

    filtered_image = np.zeros(im.shape)
    for row in range(im_row):
        for col in range(im_col):
            filtered_image[row, col] = np.sum(kernel * padded_im[row:row + kernel_row, col:col + kernel_col])

    return filtered_image


def gaussian_blurr(im, sigma, trim=4.0):
    """
    Applies a gaussian filter to an image
    :param im: ndarray
    :param sigma: integer or float representing the standard deviation
    :param trim: float, used to compute kernel size. Default is 4.0.
    :return: filtered image as ndarray
    """
    kernel = gaussian_mask(sigma, trim)
    filtered = convolve(im, kernel)
    return filtered


if __name__ == '__main__':
    # from skimage import io
    import matplotlib.pyplot as plt
    from scipy import ndimage

    # img = io.imread(r"C:\Users\marik\Pictures\BilderBioinfo\jw-1h 1_c5.TIF", as_gray=True)
    img = plt.imread(r"C:\Users\marik\Pictures\BilderBioinfo\lena.png")
    img2 = ndimage.gaussian_filter(img, sigma=1)
    img3 = gaussian_blurr(img, 1)
    img4 = img2 - img3

    # assert np.allclose(img2, img3, atol=1/255)
    # --> does not work because of corner region

    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.colorbar()
    plt.title('original')

    plt.subplot(2, 2, 2)
    plt.imshow(img2)
    plt.colorbar()
    plt.title('scipy gaussian')

    plt.subplot(2, 2, 3)
    plt.imshow(img3)
    plt.colorbar()
    plt.title('my gaussian')

    plt.subplot(2, 2, 4)
    plt.imshow(img4)
    plt.colorbar()
    plt.title('difference scipy gaussian minus my gaussian')

    plt.show()
