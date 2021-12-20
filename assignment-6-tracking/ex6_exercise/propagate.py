import numpy as np

#TODO
def propagate(particles, frame_height, frame_width, params):

    A = np.zeros((2, 2))

    if (params["model"] == 0):
        A = np.random.normal(loc=0.0, scale=1.0, size=(2, 2)) # Noise only
    elif (params["model"] == 1):
        A = np.random.normal(loc=0.0, scale=1.0, size=(2, 2)) # Constant velocity

    #to avoid dingeries for now
    A = np.abs(A)

    #multiply after the check ?
    particles = particles @ A.T

    particles_x = particles[:, 0]
    particles_filter_mask_x = np.logical_and(particles_x >= 0, particles_x < frame_width)

    particles_y = particles[:, 1]
    particles_filter_mask_y = np.logical_and(particles_y >= 0, particles_y < frame_height)

    mask = np.logical_and(particles_filter_mask_x, particles_filter_mask_y)

    mask_indices = np.where(mask)
    #particles = particles[mask_indices]

    return particles