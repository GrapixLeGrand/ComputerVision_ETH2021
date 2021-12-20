import numpy as np

#TODO
def resample(particles, particles_w):

    particles_w_p  = particles_w.reshape(particles_w.shape[0])
    particles_w_p /= np.sum(particles_w_p)

    result_indices = np.random.choice(np.arange(len(particles)), replace=True, p=particles_w_p)
    return particles[result_indices], particles_w[result_indices]