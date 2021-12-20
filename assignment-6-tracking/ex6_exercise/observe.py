
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
#TODO 
def observe(particles, frame, bbox_height, bbox_width, params_hist_bin, target_hist, params_sigma_observe):

    n_particles, _ = particles.shape
    particles_w = np.zeros(n_particles)
    box_half_dim = np.array([bbox_width // 2, bbox_height // 2])
    w, h, _ = frame.shape

    for i in range(0, n_particles):

        pi = particles[i]

        top_left = (pi - box_half_dim).astype(int)
        bottom_right = (pi + box_half_dim).astype(int)

        top_left = np.clip(top_left , 0, w)
        bottom_right = np.clip(bottom_right, 0, h)

        histogram = color_histogram(top_left[0], top_left[1], bottom_right[0], bottom_right[1], frame, params_hist_bin)
        chi2 = chi2_cost(histogram, target_hist)
        particles_w[i] = (1.0 / (params_sigma_observe * ((2*np.pi)**0.5))) * np.exp(-((chi2)**2)/ (2 * params_sigma_observe**2))

    return particles_w[:, None]