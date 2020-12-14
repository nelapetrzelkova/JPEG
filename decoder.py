import numpy as np
from PIL import Image
import os
from utils import *

RATIO_BITS = 4
WIDTH_BITS = 16

TABLE_SIZE_BITS = 16
L_BLOCKS_COUNT_BITS = 32
C_BLOCKS_COUNT_BITS = 32

DC_CODE_LENGTH_BITS = 4
CATEGORY_BITS = 4

AC_CODE_LENGTH_BITS = 8
RUN_LENGTH_BITS = 4
SIZE_BITS = 4

# constants for transforming RGB to Y'CbCr
Kr = 0.299  # Kr + Kg + Kb = 1
Kg = 0.587
Kb = 0.114

BLOCK_SIZE = 8

luma = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                 [12, 12, 14, 19, 26, 58, 60, 55],
                 [14, 13, 16, 24, 40, 57, 69, 56],
                 [14, 17, 22, 29, 51, 87, 80, 62],
                 [18, 22, 37, 56, 68, 109, 103, 77],
                 [24, 35, 55, 64, 81, 104, 113, 92],
                 [49, 64, 78, 87, 103, 121, 120, 101],
                 [72, 92, 95, 98, 112, 100, 103, 99]]).astype(np.int32)

chroma = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                   [18, 21, 26, 66, 99, 99, 99, 99],
                   [24, 26, 56, 99, 99, 99, 99, 99],
                   [47, 66, 99, 99, 99, 99, 99, 99],
                   [99, 99, 99, 99, 99, 99, 99, 99],
                   [99, 99, 99, 99, 99, 99, 99, 99],
                   [99, 99, 99, 99, 99, 99, 99, 99],
                   [99, 99, 99, 99, 99, 99, 99, 99]]).astype(np.int32)

# luma = np.array([[2, 2, 2, 2, 3, 4, 5, 6],
#                       [2, 2, 2, 2, 3, 4, 5, 6],
#                       [2, 2, 2, 2, 4, 5, 7, 9],
#                       [2, 2, 2, 4, 5, 7, 9, 12],
#                       [3, 3, 4, 5, 8, 10, 12, 12],
#                       [4, 4, 5, 7, 10, 12, 12, 12],
#                       [5, 5, 7, 9, 12, 12, 12, 12],
#                       [6, 6, 9, 12, 12, 12, 12, 12]]).astype(np.int32)
#
# chroma = np.array([[3, 3, 5, 9, 13, 15, 15, 15],
#                       [3, 4, 6, 11, 14, 12, 12, 12],
#                       [5, 6, 9, 14, 12, 12, 12, 12],
#                       [9, 11, 14, 12, 12, 12, 12, 12],
#                       [13, 14, 12, 12, 12, 12, 12, 12],
#                       [15, 12, 12, 12, 12, 12, 12, 12],
#                       [15, 12, 12, 12, 12, 12, 12, 12],
#                       [15, 12, 12, 12, 12, 12, 12, 12]])

zigzag_idxs = np.array(
        [0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14,
         21, 28, 35,
         42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47,
         55, 62, 63])



def read_dc_table(f):
    table = dict()
    t_size = read(f, TABLE_SIZE_BITS)
    for _ in range(t_size):
        category = read(f, CATEGORY_BITS)
        length = read(f, DC_CODE_LENGTH_BITS)
        code = read_str(f, length)
        table[code] = category
    return table


def read_ac_table(f):
    table = dict()
    t_size = read(f, TABLE_SIZE_BITS)
    for _ in range(t_size):
        zeros_length = read(f, RUN_LENGTH_BITS)
        size = read(f, SIZE_BITS)
        length = read(f, AC_CODE_LENGTH_BITS)
        code = read_str(f, length)
        table[code] = (zeros_length, size)
    return table


def decode_huffman(f, table):
    prefix = ''
    while prefix not in table:
        prefix += read_str(f, 1)
    return table[prefix]


def read_file(path):
    with open(path, "r") as f:
        width = read(f, WIDTH_BITS)
        ratio = (read(f, RATIO_BITS), read(f, RATIO_BITS))

        tables = dict()
        for table_name in ['dc_y', 'ac_y', 'dc_c', 'ac_c']:
            if 'dc' in table_name:
                tables[table_name] = read_dc_table(f)
            else:
                tables[table_name] = read_ac_table(f)

        l_blocks_count = read(f, L_BLOCKS_COUNT_BITS)
        c_blocks_count = read(f, C_BLOCKS_COUNT_BITS)
        dc_l = np.zeros((l_blocks_count, 1)).astype(int)
        ac_l = np.zeros((l_blocks_count, 63, 1)).astype(int)
        dc_c = np.zeros((c_blocks_count, 2)).astype(int)
        ac_c = np.zeros((c_blocks_count, 63, 2)).astype(int)

        for channel in range(3):
            if channel == 0:
                dc_table = tables['dc_y']
                ac_table = tables['ac_y']
                blocks_count = l_blocks_count
            else:
                dc_table = tables['dc_c']
                ac_table = tables['ac_c']
                blocks_count = c_blocks_count
            for block in range(blocks_count):
                # DC table decoding
                category = decode_huffman(f, dc_table)
                if channel == 0:
                    dc_l[block, channel] = read_int(f, category)
                else:
                    dc_c[block, channel-1] = read_int(f, category)

                # AC table decoding
                cells_count = 0
                while cells_count < 63:
                    zeros_length, size = decode_huffman(f, ac_table)

                    if (zeros_length, size) == (0, 0):
                        while cells_count < 63:
                            if channel == 0:
                                ac_l[block, cells_count, channel] = 0
                            else:
                                ac_c[block, cells_count, channel-1] = 0
                            cells_count += 1
                    else:
                        if channel == 0:
                            ac_l[block, cells_count, channel] = 0
                        else:
                            ac_c[block, cells_count, channel-1] = 0
                        cells_count += zeros_length
                        if size != 0:
                            val = read_int(f, size)
                            if channel == 0:
                                ac_l[block, cells_count, channel] = val
                            else:
                                ac_c[block, cells_count, channel-1] = val
                        cells_count += 1

    coeffs = dc_l, ac_l, dc_c, ac_c
    return coeffs, tables, width, ratio


def dequantize(blocks):
    quantized = []
    for i, channel in enumerate(blocks):
        quantized_channel = []
        for block in channel:
            if i == 0:  # Y channel - we use luma quantization table
                block *= luma
            else:  # Cb or Cr channel - we use chroma quantization table
                block *= chroma
            quantized_channel.append(block)
        quantized.append(quantized_channel)
    return quantized


def upsample(blocks, h, w, ratio=(2, 2)):
    rounded_w = int(np.ceil(w/BLOCK_SIZE)*BLOCK_SIZE)
    rounded_h = int(np.ceil(h/BLOCK_SIZE)*BLOCK_SIZE)
    new_img = np.zeros((rounded_h, rounded_w, 3))
    for c, channel in enumerate(blocks):
        for i, block in enumerate(channel):
            if c == 0:
                y = (i*BLOCK_SIZE//rounded_w)*BLOCK_SIZE
                x = i*BLOCK_SIZE % rounded_w
                end_y = y + BLOCK_SIZE
                end_x = x + BLOCK_SIZE
                new_img[y:end_y, x:end_x, c] = block
            else:
                y = (i*(BLOCK_SIZE*ratio[0])//w)*BLOCK_SIZE*ratio[0]
                x = i*(BLOCK_SIZE*ratio[1]) % w
                end_y = y + BLOCK_SIZE*ratio[0]
                end_x = x + BLOCK_SIZE*ratio[1]
                block = np.repeat(np.repeat(block, ratio[0], axis=0), ratio[1], axis=1)
                if end_y > rounded_h:
                    block = block[:rounded_h-end_y, :]
                if end_x > rounded_w:
                    block = block[:, :rounded_w-end_x]
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


def zigzag_to_block(zigzag):
    # assuming that the width and the height of the block are equal

    block = np.zeros((BLOCK_SIZE*BLOCK_SIZE), np.int32)

    for i in range(BLOCK_SIZE*BLOCK_SIZE):
        idx = zigzag_idxs[i]
        block[idx] = zigzag[i]
    reshaped = block.reshape((BLOCK_SIZE, BLOCK_SIZE))
    return reshaped


def precalculate_cos():
    N = 8
    table = np.zeros((8, 8))
    for m in range(N):
        for i in range(N):
            table[m, i] = np.cos((np.pi * m * (2 * i + 1)) / (2 * N))
    return table


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


def idct(pixel_blocks):
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
            transformed_channel.append(idct_2d(block, table)+128)
        transformed.append(transformed_channel)
    return transformed


def create_blocks(coeffs):
    dc_l, ac_l, dc_c, ac_c = coeffs
    blocks = list()
    luma_blocks = list()
    chroma_r_blocks = list()
    chroma_b_blocks = list()
    for block in range(dc_l.shape[0]):
        zigzag = [dc_l[block, 0]] + list(ac_l[block, :, 0])
        quant_mat = zigzag_to_block(zigzag)
        luma_blocks.append(quant_mat)
    for block in range(dc_c.shape[0]):
        zigzag1 = [dc_c[block, 0]] + list(ac_c[block, :, 0])
        zigzag2 = [dc_c[block, 1]] + list(ac_c[block, :, 1])
        quant_mat1 = zigzag_to_block(zigzag1)
        quant_mat2 = zigzag_to_block(zigzag2)
        chroma_r_blocks.append(quant_mat1)
        chroma_b_blocks.append(quant_mat2)
    blocks.append(luma_blocks)
    blocks.append(chroma_r_blocks)
    blocks.append(chroma_b_blocks)
    return blocks

def decode_and_save(in_folder, out_folder):
    bin_files = sorted(os.listdir(in_folder))
    for file in bin_files:
        path = os.path.join(in_folder, file)
        coeffs, tables, width, ratio = read_file(path)
        dc_l, ac_l, dc_c, ac_c = coeffs
        height = dc_l.shape[0] * BLOCK_SIZE ** 2 // width
        table = precalculate_cos()
        blocks = create_blocks(coeffs)
        dequantized = dequantize(blocks)
        blocks = idct(dequantized)
        img = upsample(blocks, height, width, ratio)
        rgb_image = ycbcr2rgb(img)
        image = Image.fromarray(rgb_image[:height, :width, :], 'RGB')
        image.show()
        image.save(os.path.join(out_folder, file[:-4]))

if __name__ == "__main__":
    decode_and_save('bin_files_correct', 'jpeg_images')





