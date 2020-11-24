import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.util import view_as_windows
import copy

# constants for transforming RGB to Y'CbCr
Kr = 0.299  # Kr + Kg + Kb = 1
Kg = 0.587
Kb = 0.114

# quantization matrices for quality of 50% (https://arxiv.org/pdf/1405.6147.pdf)
luma_quantization_table = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                    [12, 12, 14, 19, 26, 58, 60, 55],
                                    [14, 13, 16, 24, 40, 57, 69, 56],
                                    [14, 17, 22, 29, 51, 87, 80, 62],
                                    [18, 22, 37, 56, 68, 109, 103, 77],
                                    [24, 35, 55, 64, 81, 104, 113, 92],
                                    [49, 64, 78, 87, 103, 121, 120, 101],
                                    [72, 92, 95, 98, 112, 100, 103, 99]]).astype(np.int32)

chroma_quantization_table = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                                      [18, 21, 26, 66, 99, 99, 99, 99],
                                      [24, 26, 56, 99, 99, 99, 99, 99],
                                      [47, 66, 99, 99, 99, 99, 99, 99],
                                      [99, 99, 99, 99, 99, 99, 99, 99],
                                      [99, 99, 99, 99, 99, 99, 99, 99],
                                      [99, 99, 99, 99, 99, 99, 99, 99],
                                      [99, 99, 99, 99, 99, 99, 99, 99]]).astype(np.int32)

zigzag_idxs = np.array([0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, 35,
                        42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63])

# just a block to check correctness of DCT and quantization
test_block = np.array([[-41, -33, -58, 35, 58, -51, -15, -12],
                       [5, -34, 49, 18, 27, 1, -5, 3],
                       [-46, 14, 80, -35, -50, 19, 7, -18],
                       [-53, 21, 34, -20, 2, 34, 36, 12],
                       [9, -2, 9, -5, -32, -15, 45, 37],
                       [-8, 15, -16, 7, -8, 11, 4, 7],
                       [19, -28, -2, -26, -2, 7, -44, -21],
                       [18, 25, -12, -44, 35, 48, -37, -3]])


def rgb2ycbcr(img):
    """
    Transforms image from RGB(A) color space to Y'CbCr color space
    :param img: image in RGB(A), np array (h, w, ch) (ch=3 -> RGB, ch=4 -> RGBA)
    :return: image in Y'CbCr (h, w, 3)
    """
    if img.shape[2] == 4:  # let's ignore the alpha values
        img = img[:, :, 0:3]

    coefficients = np.array([[Kr, Kg, Kb],  # https://en.wikipedia.org/wiki/YCbCr#YCbCr
                             [-0.5 * Kr / (1 - Kb), -0.5 * Kg / (1 - Kb), 0.5],
                             [0.5, -0.5 * Kg / (1 - Kr), -0.5 * Kb / (1 - Kr)]])
    YCbCr = np.dot(img, coefficients.T)
    YCbCr[:, :, 1:] += 128
    return np.uint8(YCbCr)


#
#
def ycbcr2rgb(img):
    """
    Transform image in Y'CbCr color space to RGB color space
    :param img: image in Y'CbCr, np array (h, w, 3)
    :return: image in RGB color space, np array (h, w, 3)
    """
    coefficients = np.array([[1, 0, 2 - 2 * Kr],  # https://en.wikipedia.org/wiki/YCbCr#YCbCr
                             [1, -(Kb / Kg) * (2 - 2 * Kb), -(Kr / Kg) * (2 - 2 * Kr)],
                             [1, 2 - 2 * Kb, 0]])

    img = img.astype(np.float)
    img[:, :, 1:] -= 128
    rgb = np.dot(img, coefficients.T)
    rgb = np.where(rgb > 255, 255, rgb)
    rgb = np.where(rgb < 0, 0, rgb)
    return np.uint8(rgb)


def downsample(img, ratio=(2, 2), scikit=True):
    """
    Downsample the chroma channels of given image
    :param img: image in Y'CbCr color space, np array (h, w, ch)
    :param ratio: tuple indicating factor of downsampling in x and y directions
    :param scikit: use scikit library or for loops for downsampling, bool
    :return: list of np channels, where chroma channels are downsampled
    """

    h, w = img.shape[0], img.shape[1]
    # img = np.ones((h+1, w+1, 3))  # try image dimensions that are not divisible by downsmapling ratio
    downsampled = []
    Y = img[:, :, 0]
    Cb = img[:, :, 1]
    Cr = img[:, :, 2]

    shape = (int(h / ratio[0] + h % ratio[0]), int(w / ratio[1] + w % ratio[1]))

    if not scikit:  # slower version with for loops
        downsampled_Cb = np.zeros(shape)
        downsampled_Cr = np.zeros_like(downsampled_Cb)
        for i in range(0, h, ratio[0]):
            for j in range(0, w, ratio[1]):
                # do the downsampling by averaging the value of the pixels and rounding to the nearest integer
                downsampled_Cb[i // 2, j // 2] = np.round(np.mean(Cb[i: i + ratio[0], j: j + ratio[1]])).astype(int)
                downsampled_Cr[i // 2, j // 2] = np.round(np.mean(Cr[i: i + ratio[0], j: j + ratio[1]])).astype(int)

    else:  # vectorized, hence much faster version of downsampling with skicit-image library
        Cb_windows = view_as_windows(Cb, ratio, ratio)
        Cr_windows = view_as_windows(Cr, ratio, ratio)
        downsampled_Cb = np.round(np.mean(Cb_windows, axis=(2, 3))).astype(int)
        downsampled_Cr = np.round(np.mean(Cr_windows, axis=(2, 3))).astype(int)

    downsampled.extend((Y, downsampled_Cb, downsampled_Cr))

    return downsampled


def create_pixel_groups(img):
    """
    Create list of blocks of size block_size x block_size from given image. pAdding is applied if necessary
    :param img: list (3) of channels, due to downsampling not all the channels have the same size
    :return: list of blocks shaped (8, 8, 3)
    """

    block_size = 8  # not as changeable parameter because our quantization tables are 8 x 8

    for i, channel in enumerate(img):
        h, w = channel.shape
        # padding at the end of image, we copy the values of last row/column in image remainder-times
        if h % block_size != 0:
            remainder = h % block_size
            last_row = channel[-1, :]
            to_add = np.tile(last_row, (remainder, 1))
            channel = np.concatenate((channel, to_add), axis=0)
        if w % block_size != 0:
            remainder = w % block_size
            last_col = channel[:, -1]
            to_add = np.tile(last_col[:, np.newaxis], (1, remainder))
            channel = np.concatenate((channel, to_add), axis=1)
        img[i] = channel

    pixel_groups = []
    for channel in img:
        h, w = channel.shape
        pixel_block = []
        for i in range(int(h / block_size)):
            for j in range(int(w / block_size)):  # loop over the image and create the groups
                pixel_block.append(channel[i * block_size:(i + 1) * block_size, j * block_size: (j + 1) * block_size])
        pixel_groups.append(pixel_block)

    return pixel_groups


def precalculate_cos():
    N = 8
    table = np.zeros((8,8))
    for m in range(N):
        for i in range(N):
            table[m, i] = np.cos((np.pi * m * (2 * i + 1)) / (2 * N))
    return table


def dct_2d(pixel_blocks, table):
    step1 = np.apply_along_axis(dct_1d, axis=0, arr=pixel_blocks, table=table)
    step2 = np.apply_along_axis(dct_1d, axis=1, arr=step1, table=table)
    return step2


def dct_1d(pixel_blocks, table):
    N = pixel_blocks.shape[0]  # 8
    coefficients = np.zeros(N)
    for m in range(N):
        for i in range(N):
            coefficients[m] += pixel_blocks[i] * table[m, i]
        alpha = 1/np.sqrt(2) if m == 0 else 1
        coefficients[m] *= alpha
    return coefficients/2


def idct_2d(pixel_blocks, table):
    step1 = np.apply_along_axis(idct_1d, axis=0, arr=pixel_blocks, table=table)
    step2 = np.apply_along_axis(idct_1d, axis=1, arr=step1, table=table)
    return step2


def idct_1d(pixel_blocks, table):
    N = pixel_blocks.shape[0]  # 8
    coefficients = np.zeros(N)
    for m in range(N):
        for i in range(N):
            coefficients[m] += pixel_blocks[i] * table[i,m]
            alpha = 1/np.sqrt(2) if i == 0 else 1
            coefficients[m] *= alpha
    return coefficients/2


def discrete_cosine_transform(pixel_blocks, inverse=False):
    """
    Applies discrete cosine transform to pixel groups (8 x 8) in each channel
    :param pixel_blocks: list (3) of lists of blocks (8 x 8)
    :return: blocks with DCT values, same shape like pixel_blocks
    """
    transformed = []
    table = precalculate_cos()
    for channel in pixel_blocks:  # for each channel
        transformed_channel = []
        for block in channel:  # for each pixel block (8 x 8)
            if not inverse:
                shifted = block.astype(int)-128
                transformed_channel.append(dct_2d(shifted, table))  # perform DCT
            else:
                transformed_channel.append(idct_2d(block, table)+128)
        transformed.append(transformed_channel)
    return transformed


def quantize(blocks):
    """
    Quantize pixel blocks with given quantization tables
    :param blocks: list of 8x8 blocks of transformed values by DCT
    :return: list of quantized blocks
    """
    luma = luma_quantization_table
    chroma = chroma_quantization_table
    quantized = []
    for i, channel in enumerate(blocks):
        quantized_channel = []
        for block in channel:
            if i == 0:  # Y channel - we use luma quantization table
                block = np.round(block / luma)
            else:  # Cb or Cr channel - we use chroma quantization table
                block = np.round(block / chroma)
            quantized_channel.append(block)
        quantized.append(quantized_channel)
    return quantized


def dequantize(blocks):
    dequantized = []
    luma = luma_quantization_table
    chroma = chroma_quantization_table
    for i, channel in enumerate(blocks):
        quantized_channel = []
        for block in channel:
            if i == 0:  # Y channel - we use luma quantization table
                block *= luma
            else:  # Cb or Cr channel - we use chroma quantization table
                block *= chroma
            quantized_channel.append(block)
        dequantized.append(quantized_channel)
    return dequantized


def upsample(blocks, image, ratio=(2,2)):
    block_size = 8
    h, w, c = image.shape
    print(image.shape)
    rounded_w = int(np.ceil(w/block_size)*block_size)
    rounded_h = int(np.ceil(h/block_size)*block_size)
    print(rounded_h, rounded_w)
    new_img = np.zeros((rounded_h, rounded_w, c))
    for c, channel in enumerate(blocks):
        for i, block in enumerate(channel):
            if c == 0:
                y = (i*block_size//w)*block_size
                x = i*block_size%w
                end_y = y + block_size
                end_x = x + block_size
                new_img[y:end_y, x:end_x, c] = block
            else:
                y = (i*(block_size*ratio[0])//w)*block_size*ratio[0]
                x = i*(block_size*ratio[1])%w
                end_y = y + block_size*ratio[0]
                end_x = x + block_size*ratio[1]
                block = np.repeat(np.repeat(block, 2, axis=0), 2, axis=1)
                new_img[y:end_y, x:end_x, c] = block
    return new_img


def ycbcr2rgb(img):
    """
    Transform image in Y'CbCr color space to RGB color space
    :param img: image in Y'CbCr, np array (h, w, 3)
    :return: image in RGB color space, np array (h, w, 3)
    """
    coefficients = np.array([[1, 0, 2 - 2 * Kr],  # https://en.wikipedia.org/wiki/YCbCr#YCbCr
                             [1, -(Kb / Kg) * (2 - 2 * Kb), -(Kr / Kg) * (2 - 2 * Kr)],
                             [1, 2 - 2 * Kb, 0]])

    img = img.astype(np.float)
    img[:, :, 1:] -= 128
    rgb = np.dot(img, coefficients.T)
    rgb = np.where(rgb > 255, 255, rgb)
    rgb = np.where(rgb < 0, 0, rgb)
    return np.uint8(rgb)


if __name__ == '__main__':
    image = Image.open('sample.bmp')
    image = np.array(image)
    plt.imshow(image)
    plt.show()
    ycbcr = rgb2ycbcr(image)
    plt.imshow(ycbcr)
    plt.show()
    downsampled = downsample(ycbcr)
    pixel_groups = create_pixel_groups(downsampled)
    transformed = discrete_cosine_transform(pixel_groups)
    quantized = quantize(transformed)
    dequantized = dequantize(copy.deepcopy(quantized))
    idct = discrete_cosine_transform(dequantized, inverse=True)
    upsampled = upsample(idct, image)
    rgb = ycbcr2rgb(upsampled)
    plt.imshow(rgb)
    plt.show()
    pass