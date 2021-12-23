import numpy as np

#TODO
def color_histogram(xmin, ymin, xmax, ymax, frame, hist_bin):

    sub_frame = frame[ymin:ymax, xmin:xmax]

    hist_r, _ = np.histogram(sub_frame[:, :, 0], bins=hist_bin, density=True)
    hist_g, _ = np.histogram(sub_frame[:, :, 1], bins=hist_bin, density=True)
    hist_b, _ = np.histogram(sub_frame[:, :, 2], bins=hist_bin, density=True)

    s1 = np.sum(hist_r)
    s2 = np.sum(hist_g)
    s3 = np.sum(hist_b)
    """
    if (np.sum(hist_r) > 0):
        hist_r /= np.sum(hist_r)
    if (np.sum(hist_g) > 0):
        hist_g /= np.sum(hist_g)
    if (np.sum(hist_b) > 0):
        hist_b /= np.sum(hist_b)
    """
    hist = np.zeros((3, hist_bin))

    hist[0] = hist_r
    hist[1] = hist_g
    hist[2] = hist_b

    return hist