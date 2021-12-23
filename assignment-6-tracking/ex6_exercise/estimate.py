import numpy as np

#TODO
def estimate(particles, particles_w):
    p = particles * particles_w
    estimation = np.sum(p, axis=0)
    return estimation