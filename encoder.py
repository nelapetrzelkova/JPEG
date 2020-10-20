import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.util import view_as_windows
from scipy import fftpack
from huffman import *

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
test_block = np.array([[-415, -33, -58, 35, 58, -51, -15, -12],
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


def dct_2d(pixel_blocks):
    step1 = np.apply_along_axis(dct_1d, axis=0, arr=pixel_blocks)
    step2 = np.apply_along_axis(dct_1d, axis=1, arr=step1)
    return step2


def dct_1d(pixel_blocks):
    N = pixel_blocks.shape[0]
    coefficents = np.zeros(N)
    for m in range(N):
        for i in range(N):
            coefficents[m] += pixel_blocks[i] * np.cos((np.pi * m * (2 * i + 1)) / (2 * N))
    return coefficents


def discrete_cosine_transform(pixel_blocks):
    """
    Applies discrete cosine transform to pixel groups (8 x 8) in each channel
    :param pixel_blocks:
    :return: list (3)
    """
    transformed = []
    for channel in pixel_blocks:  # for each channel
        transformed_channel = []
        for block in channel:  # for each pixel block (8 x 8)
            transformed_channel.append(dct_2d(block))  # perform DCT
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
                block = np.floor(block / luma) * luma
            else:  # Cb or Cr channel - we use chroma quantization table
                block = np.floor(block / chroma) * chroma
            quantized_channel.append(block)
        quantized.append(quantized_channel)
    return quantized


def block_to_zigzag(block):
    return np.array([block.ravel()[idx] for idx in zigzag_idxs]).astype(int)


def invert_bin_str(binstr):
    # check if binstr is a binary string
    return ''.join(map(lambda c: '0' if c == '1' else '1', binstr))


def int_to_binary_string(n):
    if n == 0:
        return ''
    binstr = bin(abs(n))[2:]
    # change every 0 to 1 and vice verse when n is negative
    return binstr if n > 0 else invert_bin_str(binstr)


def bits_count(n):
    n = abs(int(n))
    result = 0
    while n > 0:
        n >>= 1
        result += 1
    return result


def flatten(lst):
    return [item for sublist in lst for item in sublist]


def encode_ac(arr):
    last_nonzero = -1
    for idx, elem in enumerate(arr):  # find last non-zero element in zigzag array
        if elem != 0:
            last_nonzero = idx
    symbols = []
    values = []
    zeros_length = 0
    for idx, elem in enumerate(arr):
        if idx > last_nonzero:
            symbols.append((0, 0))
            bin_str = int_to_binary_string(0)
            values.append(bin_str)
            break
        elif elem == 0:
            zeros_length += 1
        else:
            size = bits_count(elem)
            symbols.append((zeros_length, size))
            bin_str = int_to_binary_string(elem)
            values.append(bin_str)
            zeros_length = 0
    return symbols, values


def create_huffman_tables(dc, ac, blocks_count):
    dc_l = HuffmanTree(np.vectorize(bits_count)(dc[:, 0]))
    dc_c = HuffmanTree(np.vectorize(bits_count)(dc[:, 1:].flat))
    luma_symbols, _ = encode_ac(ac[i, :, 0] for i in range(blocks_count))
    chroma_symbols, _ = encode_ac(ac[i, :, j] for i in range(blocks_count) for j in [1, 2])
    ac_l = HuffmanTree(flatten(luma_symbols))
    ac_c = HuffmanTree(flatten(chroma_symbols))


def encode(pixel_blocks):
    """
    Create a Huffman encoded bit stream in a zig-zag manner
    :param pixel_blocks:
    :return:
    """
    blocks_count = len(pixel_blocks)
    dc = np.zeros((blocks_count, 3))
    ac = np.zeros((blocks_count, 63, 3))
    block_idx = 0

    for i, channel in enumerate(pixel_blocks):
        for block in channel:
            zigzag_arr = block_to_zigzag(block)  # transform block to array in a zigzag way
            dc[block_idx, i] = zigzag_arr[0]
            ac[block_idx, :, i] = zigzag_arr[1:]
            block_idx += 1

    # tables = create_huffman_tables(dc, ac, blocks_count)
    # create_encoded_file(dc, ac, blocks_count, tables)

# TODO: 3D scatter plot of Y'CbCr features
# TODO: plot every channel in Y'CbCr separately
if __name__ == '__main__':
    # image = Image.open('flag.bmp')
    # orig = np.array(image)
    # plt.imshow(orig)
    # plt.show()
    # ycbcr = rgb2ycbcr(orig)
    # plt.imshow(ycbcr)
    # plt.show()
    # ycbcr = downsample(ycbcr)
    # pixel_groups = create_pixel_groups(ycbcr)
    # transformed = discrete_cosine_transform(pixel_groups)
    # quantized = quantize(transformed)
    q = quantize(discrete_cosine_transform(list([test_block[np.newaxis, :]])))
    encode(q)
    pass