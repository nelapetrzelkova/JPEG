import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.util import view_as_windows
from scipy import fftpack
from huffman import *

TABLE_SIZE_BITS = 16
BLOCKS_COUNT_BITS = 32

DC_CODE_LENGTH_BITS = 4
CATEGORY_BITS = 4

AC_CODE_LENGTH_BITS = 8
RUN_LENGTH_BITS = 4
SIZE_BITS = 4

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




def read(f, len):
    return int(f.read(len), 2)


def read_dc_table(f):
    table = dict()
    t_size = read(f, TABLE_SIZE_BITS)
    for _ in range(t_size):
        category = read(f, CATEGORY_BITS)
        length = read(f, DC_CODE_LENGTH_BITS)
        code = read(f, length)
        table[code] = category
    return table


def read_ac_table(f):
    table = dict()
    t_size = read(f, TABLE_SIZE_BITS)
    for _ in range(t_size):
        zeros_length = read(f, RUN_LENGTH_BITS)
        size = read(f, SIZE_BITS)
        length = read(f, AC_CODE_LENGTH_BITS)
        code = read(f, length)
        table[code] = (zeros_length, size)
    return table


def decode_huffman(f, table):
    prefix = ''
    while prefix not in table:
        prefix += f.read(1)
    return table[prefix]


def read_file(path):
    with open(path, "r") as f:
        tables = dict()
        for table_name in ['dc_y', 'ac_y', 'dc_c', 'ac_c']:
            if 'dc' in table_name:
                tables[table_name] = read_dc_table(f)
            else:
                tables[table_name] = read_ac_table(f)

        blocks_count = read(f, BLOCKS_COUNT_BITS)
        dc = np.zeros((blocks_count, 3)).astype(int)
        ac = np.zeros((blocks_count, 63, 3)).astype(int)

        for block in range(blocks_count):
            for channel in range(3):
                if channel == 0:
                    dc_table = tables['dc_y']
                    ac_table = tables['ac_y']
                else:
                    dc_table = tables['dc_c']
                    ac_table = tables['ac_c']
                category = decode_huffman(f, dc_table)
                dc[block, channel] =


def dequantize(blocks):
    dequantized = []
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
    rounded_w = int(np.ceil(w/block_size)*block_size)
    rounded_h = int(np.ceil(h/block_size)*block_size)
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

if __name__ == "__main__":
    read_file('out')