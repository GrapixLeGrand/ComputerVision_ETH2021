import numpy as np

#TODO
def propagate(particles, frame_height, frame_width, params):

    if (params["model"] == 0):
        noise = np.random.standard_normal(particles.shape)
        std_dev = params['sigma_position']
        particles += noise * std_dev
    elif (params["model"] == 1):
        noise = np.random.standard_normal(particles.shape)
        std_dev = params['sigma_velocity']
        
        A = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        #particles @ A.T + 
        particles = particles @ A.T 
        particles += noise * std_dev

    # if the particles move out of the box dont remove them, simply clamp them
    particles[:, 0] = np.clip(particles[:, 0], 0, frame_width - 1)
    particles[:, 1] = np.clip(particles[:, 1], 0, frame_height - 1)

    return particles