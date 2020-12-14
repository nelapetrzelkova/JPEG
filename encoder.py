import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.util import view_as_windows
from huffman import *
import os
from utils import *

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

zigzag_idxs = np.array(
    [0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21,
     28, 35,
     42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55,
     62, 63])

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
            to_add = np.tile(last_row, (block_size-remainder, 1))
            channel = np.concatenate((channel, to_add), axis=0)
        if w % block_size != 0:
            remainder = w % block_size
            last_col = channel[:, -1]
            to_add = np.tile(last_col[:, np.newaxis], (1, block_size-remainder))
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
        alpha = 1 / np.sqrt(2) if m == 0 else 1
        coefficients[m] *= alpha
    return coefficients / 2


def discrete_cosine_transform(pixel_blocks):
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
            shifted = block.astype(int) - 128
            transformed_channel.append(dct_2d(shifted, table))  # perform DCT
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
                block = np.trunc(block / luma)
            else:  # Cb or Cr channel - we use chroma quantization table
                block = np.trunc(block / chroma)
            quantized_channel.append(block)
        quantized.append(quantized_channel)
    return quantized


def block_to_zigzag(block):
    return np.array([block.ravel()[idx] for idx in zigzag_idxs]).astype(int)


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
            bin_str = int_to_binary_string(int(elem))
            values.append(bin_str)
            zeros_length = 0
    return symbols, values


def create_huffman_tables(coeffs):
    dc_l, ac_l, dc_c, ac_c = coeffs
    dc_l = HuffmanTree(np.vectorize(bits_count)(dc_l[:, 0]))
    dc_c = HuffmanTree(np.vectorize(bits_count)(dc_c.flat))
    ac_l = HuffmanTree(flatten(encode_ac(ac_l[i, :, 0])[0]
                               for i in range(ac_l.shape[0])))
    ac_c = HuffmanTree(flatten(encode_ac(ac_c[i, :, j])[0]
                               for i in range(ac_c.shape[0]) for j in [0, 1]))
    tables = {'dc_l': dc_l.value_to_bitstring_table(),
              'ac_l': ac_l.value_to_bitstring_table(),
              'dc_c': dc_c.value_to_bitstring_table(),
              'ac_c': ac_c.value_to_bitstring_table()}
    return tables


def create_encoded_file(coeffs, tables, ratio, width, filename="out"):
    """
    Save binary data to a file named 'out'
    :param dc: data from the first element in the blocks (intensity)
    :param ac: data from the all other elements in the blocks
    :param tables: huffman tables
    :return:
    """
    dc_l, ac_l, dc_c, ac_c = coeffs
    l_blocks_count = dc_l.shape[0]
    c_blocks_count = dc_c.shape[0]
    with open(filename, "w+") as f:
        f.write(uint_to_binstr(width, 16))
        f.write(uint_to_binstr(ratio[0], 4))
        f.write(uint_to_binstr(ratio[1], 4))
        for idx, table in enumerate(tables.values()):
            n = uint_to_binstr(len(table), 16)
            f.write(n)
            for key, val in table.items():
                if idx == 0 or idx == 2:  # dc tables
                    f.write(uint_to_binstr(key, 4))
                    f.write(uint_to_binstr(len(val), 4))
                    f.write(val)
                else:  # ac tables
                    f.write(uint_to_binstr(key[0], 4))
                    f.write(uint_to_binstr(key[1], 4))
                    f.write(uint_to_binstr(len(val), 8))
                    f.write(val)
        f.write(uint_to_binstr(l_blocks_count, 32))
        f.write(uint_to_binstr(c_blocks_count, 32))
        for channel in range(3):
            if channel == 0:
                for block in range(dc_l.shape[0]):
                    bits = bits_count(dc_l[block, 0])
                    symbols, values = encode_ac(ac_l[block, :, 0])
                    dc_table = tables['dc_l']
                    f.write(dc_table[bits])
                    f.write(int_to_binary_string(int(dc_l[block, channel])))

                    ac_table = tables['ac_l']
                    for i in range(len(symbols)):
                        f.write(ac_table[tuple(symbols[i])])
                        f.write(values[i])
            else:
                for block in range(dc_c.shape[0]):
                    bits = bits_count(dc_c[block, channel - 1])
                    symbols, values = encode_ac(ac_c[block, :, channel - 1])
                    dc_table = tables['dc_c']
                    f.write(dc_table[bits])
                    f.write(int_to_binary_string(int(dc_c[block, channel - 1])))

                    ac_table = tables['ac_c']
                    for i in range(len(symbols)):
                        f.write(ac_table[tuple(symbols[i])])
                        f.write(values[i])


def encode(pixel_blocks, ratio, width, filename):
    dc_l = np.zeros((len(pixel_blocks[0]), 1))
    ac_l = np.zeros((len(pixel_blocks[0]), 63, 1))
    dc_c = np.zeros((len(pixel_blocks[1]), 2))
    ac_c = np.zeros((len(pixel_blocks[1]), 63, 2))

    for i, channel in enumerate(pixel_blocks):

        block_idx = 0
        for block in channel:
            zigzag_arr = block_to_zigzag(block)  # transform block to array in a zigzag way
            if i == 0:
                dc_l[block_idx, i] = zigzag_arr[0]
                ac_l[block_idx, :, i] = zigzag_arr[1:]
                block_idx += 1
            else:
                dc_c[block_idx, i - 1] = zigzag_arr[0]
                ac_c[block_idx, :, i - 1] = zigzag_arr[1:]
                block_idx += 1
    coeffs = dc_l, ac_l, dc_c, ac_c
    tables = create_huffman_tables(coeffs)
    create_encoded_file(coeffs, tables, ratio, width, filename)


def create_bin_files(filename, ratios, out_filename):
    image = Image.open(filename)
    orig = np.array(image)
    width = orig.shape[1]
    plt.imshow(orig)
    plt.show()
    ycbcr = rgb2ycbcr(orig)

    for ratio in ratios:
        downsampled = downsample(ycbcr, ratio)
        pixel_groups = create_pixel_groups(downsampled)
        transformed = discrete_cosine_transform(pixel_groups)
        quantized = quantize(transformed)
        out_name = os.path.join(out_filename, filename.split("/")[1][:-4] + ratio_str(ratio))
        encode(quantized, ratio, width, filename=out_name)


if __name__ == '__main__':
    img_folder = 'images'
    bin_folder = 'bin_files_correct'
    out_name = "sizes_correct.png"
    out_file = "orig_sizes.png"
    images = sorted(os.listdir(img_folder))
    ratios = [(1, 1), (2, 1), (2, 2), (4, 1), (4, 4), (6, 6), (8, 8)]
    images_names = list(map(lambda x: x[:-4], images))
    orig_size_table(img_folder, out_file, images_names)
    plot_imgs(img_folder)
    # for img in images:
    #     create_bin_files(os.path.join(img_folder, img), ratios, bin_folder)
    create_size_tables(bin_folder, out_name, ratios, img_folder, images_names)






