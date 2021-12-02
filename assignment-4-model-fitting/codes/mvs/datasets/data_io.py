import numpy as np
import re
import sys
from PIL import Image
from torch.functional import norm

def read_cam_file(filename):

    # TODO
    with open(filename, "r") as data:

        # convert list of str to list of list of str where each element is either a float or a keyword
        lines = [line.split(sep=" ") for line in data]

        extrinsics = np.zeros((4, 4))
        
        #iterate over the right parts (skip extrinsics)
        for i in range(0, 4):
            tmp = [float(lines[i + 1][k]) for k in range(0, 4)]
            extrinsics[i] = np.array(tmp)

        intrinsics = np.zeros((3, 3))

        #iterate over the right parts (skip intrisics + beginning)
        for i in range(0, 3):
            tmp = [float(lines[i + 7][k]) for k in range(0, 3)]
            intrinsics[i] = np.array(tmp)

        depth_min = float(lines[11][0])
        depth_max = float(lines[11][1])

    return intrinsics, extrinsics, depth_min, depth_max

def read_img(filename):
    # data is a (width, height, 3) array (initially 3 uint8)

    #https://stackoverflow.com/questions/31386096/importing-png-files-into-numpy
    """
    img_frame = Image.open(filename)
    data = np.array(img_frame.getdata(), dtype=np.float32) / 255.0
    if (img_frame.mode == 'RGB'):
        data = data.reshape((img_frame.width, img_frame.height, 3))
    elif(img_frame.mode == 'L'):
        data = data.reshape((img_frame.width, img_frame.height))
    else:
        assert True, "unknown mode !"
    """
    return np.asarray(Image.open(filename), dtype=np.float32) / 255.0

def read_depth(filename):
    # read pfm depth file
    return np.array(read_pfm(filename)[0], dtype=np.float32)

def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    ll = file.readline()
    print(ll)
    header = ll.decode(encoding='utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def save_pfm(filename, image, scale=1):
    file = open(filename, "wb")
    color = None

    image = np.flipud(image)

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode('utf-8') if color else 'Pf\n'.encode('utf-8'))
    file.write('{} {}\n'.format(image.shape[1], image.shape[0]).encode('utf-8'))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(('%f\n' % scale).encode('utf-8'))

    image.tofile(file)
    file.close()
