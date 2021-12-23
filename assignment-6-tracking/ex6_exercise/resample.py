import numpy as np

#TODO
def resample(particles, particles_w):

    particles_w_p  = particles_w.reshape(particles_w.shape[0])
    #particles_w_p /= np.sum(particles_w_p) # should we do this ?

    result_indices = np.random.choice(len(particles), size=(len(particles),), replace=True, p=particles_w_p)

    particles_w = particles_w[result_indices]
    w_sum = np.sum(particles_w)
    if (w_sum > 0.0):
        particles_w /= w_sum
    else:
        print("weight in resample are 0")

    assert np.allclose(np.sum(particles_w), 1), "particles weight not sum to almost 1"

    return particles[result_indices], particles_w[result_indices]