import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import six
import matplotlib.image as mpimg


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


def uint_to_binstr(number, size):
    return bin(number)[2:][-size:].zfill(size)


def read_str(f, len):
    return f.read(len)


def read(f, len):
    return int(f.read(len), 2)


def binstr_flip(binstr):
    return ''.join(map(lambda c: '0' if c == '1' else '1', binstr))


def read_int(f, size):
    if size == 0:
        return 0

    # the most significant bit indicates the sign of the number
    bin_num = read_str(f, size)
    if bin_num[0] == '1':
        return int(bin_num, 2)
    else:
        return int(binstr_flip(bin_num), 2) * -1


def precalculate_cos():
    N = 8
    table = np.zeros((8, 8))
    for m in range(N):
        for i in range(N):
            table[m, i] = np.cos((np.pi * m * (2 * i + 1)) / (2 * N))
    return table


def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax


def ratio_str(ratio):
    return "_" + str(ratio[0]) + str(ratio[1])


def plot_imgs(img_folder):
    images = sorted(os.listdir(img_folder))
    fig = plt.figure(figsize=(15, 5))
    fig.subplots(1,len(images))
    for idx, img in enumerate(images):
        plt.subplot(1, len(images), idx + 1)
        path = os.path.join(img_folder, img)
        im_arr = mpimg.imread(path)
        plt.axis('off')
        plt.title(img[:-4])
        plt.imshow(im_arr)
    fig.show()
    fig.savefig('images.png', bbox_inches='tight')


def create_size_tables(bin_folder, out_name, ratios, img_folder, images_names):
    bin_files = sorted(os.listdir(bin_folder))
    all_sizes = []
    img_sizes = []
    images = sorted(os.listdir(img_folder))
    c = 0
    for i, file in enumerate(bin_files):
        path = os.path.join(bin_folder, file)
        kb_size = os.path.getsize(path) / 1000
        perc = kb_size / os.path.getsize(os.path.join(img_folder, images[c])) * 100
        img_sizes.append("{:.2f} kB".format(kb_size))

        if i % len(ratios) == len(ratios) - 1:
            all_sizes.append(img_sizes)
            img_sizes = []
    np_sizes = np.array(all_sizes)
    str_ratios = np.array(list(map(lambda x: str(x), ratios)))
    df = pd.DataFrame(np_sizes.T)
    df.columns = images_names
    df.insert(0, "Downs. ratio", str_ratios)
    render_mpl_table(df, header_columns=0, col_width=2.2)
    plt.savefig(out_name, dpi=300, bbox_inches='tight')

def orig_size_table(img_folder, out_file, images_names):
    orig_sizes = []
    images = sorted(os.listdir(img_folder))
    for img in images:
        orig_sizes.append("{:.2f} kB".format(os.path.getsize(os.path.join(img_folder, img)) / 1000))
    np_sizes = np.array(orig_sizes)[np.newaxis, :]
    df = pd.DataFrame(np_sizes)
    df.columns = images_names
    render_mpl_table(df, header_columns=0, col_width=2.2)
    plt.savefig(out_file, dpi=300, bbox_inches='tight')