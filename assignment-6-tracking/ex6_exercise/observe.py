
import numpy as np
from color_histogram import color_histogram
from chi2_cost import chi2_cost

"""
This function should make observations i.e. compute for all particles its color histogram describing
the bounding box defined by the center of the particle and bbox height and bbox width. These
observations should be used then to update the weights particles w using eq. 6 based on the χ2
distance between the particle color histogram and the target color histogram given here as hist
target. In order to compute the χ2 distance use the provided function chi2 cost.py.
"""
# TODO


def observe(particles, frame, bbox_height, bbox_width, params_hist_bin, target_hist, params_sigma_observe):

    n_particles, _ = particles.shape
    particles_w = np.zeros(n_particles)
    #box_half_dim = np.array([bbox_width // 2, bbox_height // 2])
    frame_width, frame_height, _ = frame.shape

    for i in range(0, n_particles):

        pi = particles[i]

        #xmin = min(max(0, round(pi[0]-0.5*bbox_width)), frame_width-1)
        #xmax = min(max(0, round(pi[0]+0.5*bbox_width)), frame_width-1)

        #ymin = min(max(0, round(pi[1]-0.5*bbox_height)), frame_height-1)
        #ymax = min(max(0, round(pi[1]+0.5*bbox_height)), frame_height-1)

        #xmin = int(np.clip(pi[0] - 0.5 * bbox_width, 0.5 * bbox_width, frame_width - 0.5 * bbox_width - 1))
        #xmax = int(np.clip(pi[0] + 0.5 * bbox_width, 0.5 * bbox_width, frame_width - 0.5 * bbox_width - 1))

        xmin = int(np.clip(pi[0] - 0.5 * bbox_width, 0.0, frame_width - 1.0))
        xmax = int(np.clip(pi[0] + 0.5 * bbox_width, 0.0, frame_width - 1.0))

        if (xmax - xmin < bbox_width and xmin == 0):
            xmax = xmin + bbox_width

        if (xmax - xmin < bbox_width and xmax == frame_width - 1):
            xmin = xmax - bbox_width

        ymin = int(np.clip(pi[1] - 0.5 * bbox_height, 0.0, frame_height - 1.0))
        ymax = int(np.clip(pi[1] + 0.5 * bbox_height, 0.0, frame_height - 1.0))

        if (ymax - ymin < bbox_height and ymin == 0):
            ymax = ymin + bbox_height

        if (ymax - ymin < bbox_height and ymax == frame_height - 1):
            ymin = ymax - bbox_height

        #ymin = int(np.clip(pi[1] - 0.5 * bbox_height, 0.5 * bbox_height, frame_height - 0.5 * bbox_height - 1))
        #ymax = int(np.clip(pi[1] + 0.5 * bbox_height, 0.5 * bbox_height, frame_height - 0.5 * bbox_height - 1))

        assert xmax - xmin == bbox_width, "width of bounding box is wrong"
        assert ymax - ymin == bbox_height, "height of bounding box is wrong"
        assert xmin < xmax, "bounding box width is 0"
        assert ymin < ymax, "bounding box height is 0"
        assert xmin >= 0 and xmax <= frame_width - 1, "xmin or max outside the frame"
        assert ymin >= 0 and ymax <= frame_height - 1, "ymin or max outside the frame"

        histogram = color_histogram(
            xmin, ymin, xmax, ymax, frame, params_hist_bin)

        chi2 = chi2_cost(histogram, target_hist)
        particles_w[i] = (1.0 / (params_sigma_observe * ((2*np.pi)**0.5))) * \
            np.exp(-(chi2**2) / (2 * (params_sigma_observe**2)))

    # normalize the probabilities
    w_sum = np.sum(particles_w)
    if (w_sum > 0):
        particles_w /= w_sum

    return particles_w[:, None]
