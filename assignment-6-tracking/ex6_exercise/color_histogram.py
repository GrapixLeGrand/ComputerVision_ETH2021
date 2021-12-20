import numpy as np

#TODO
def color_histogram(xmin, ymin, xmax, ymax, frame, hist_bin):

    sub_frame = frame[xmin:xmax, ymin:ymax]
    w, h, c = sub_frame.shape
    sub_frame = sub_frame.reshape((w * h * c))
    sub_frame = sub_frame // hist_bin
    hist = np.bincount(sub_frame, minlength=hist_bin)

    return hist