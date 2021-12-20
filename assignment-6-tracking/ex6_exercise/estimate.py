import numpy as np

#TODO
def estimate(particles, particles_w):
    estimation = np.sum(particles * particles_w, axis=0)
    return estimation