import time
import os
import random
import math
import torch
import numpy as np

# run `pip install scikit-image` to install skimage, if you haven't done so
from skimage import io, color
from skimage.transform import rescale

#https://fr.wikipedia.org/wiki/%C3%89cart_de_couleur
def distance(x, X):
    #raise NotImplementedError('distance function not implemented!')
    # as seen in the link above we compute the euclidean distance between
    # the point and the rest of the image as 
    return torch.sum((X - x) ** 2, 1) ** 0.5

def distance_batch(x, X):
    #raise NotImplementedError('distance_batch function not implemented!')
    
    return torch.norm(x - X, dim=0) #torch.sum((X - x) ** 2, 1) ** 0.5

#https://en.wikipedia.org/wiki/Radial_basis_function_kernel
#looks like the bandwith is the sigma in the rbf after this post
#https://moodle-app2.let.ethz.ch/mod/forum/discuss.php?d=88657
def gaussian(dist, bandwidth):
    return torch.exp(-(dist ** 2) / (2 * bandwidth ** 2))

def update_point(weight, X):
    # unsqueeze weight (increase the whole dimension by 1: [] -> [[]])
    # then expand this dimension by 3 with same elements
    weight_ = torch.unsqueeze(weight, 0).expand(3, weight.size()[0])
    weight_ = torch.transpose(weight_, 0, 1) #swap axis 0 and 1

    #compute the shift as described here: https://en.wikipedia.org/wiki/Mean_shift
    X_weighted_sum = torch.sum(weight_ * X, axis=0)
    weight_sum = torch.sum(weight)

    result = X_weighted_sum / weight_sum
    #assert result.size() == X[0].size() # in case a ghost flattening happens
    return result

def update_point_batch(weight, X):
    print("update points X : ", X.size(), ", weight : ", weight.size())
    #raise NotImplementedError('update_point_batch function not implemented!')
    # unsqueeze weight (increase the whole dimension by 1: [] -> [[]])
    # then expand this dimension by 3 with same elements
    weight_ = torch.unsqueeze(weight, 0).expand(3, weight.size()[0])
    weight_ = torch.transpose(weight_, 0, 1) #swap axis 0 and 1

    #compute the shift as described here: https://en.wikipedia.org/wiki/Mean_shift
    X_weighted_sum = torch.sum(weight_ * X, axis=0)
    weight_sum = torch.sum(weight)

    result = X_weighted_sum / weight_sum
    #assert result.size() == X[0].size() # in case a ghost flattening happens
    return result

def meanshift_step(X, bandwidth=2.5):
    X_ = X.clone()
    for i, x in enumerate(X):
        dist = distance(x, X)
        weight = gaussian(dist, bandwidth)
        X_[i] = update_point(weight, X)
    return X_

def meanshift_step_batch(X, bandwidth=2.5):
    X = X.clone()

    dists = distance_batch(X, X)
    print("dist : ", dists.size())
    print(dists)
    weights = gaussian(dists, bandwidth)
    X_ = update_point_batch(weights, X)

    return X_

    #raise NotImplementedError('meanshift_step_batch function not implemented!')

def meanshift(X):
    X = X.clone()
    for i in range(20):
        print("iteration: ", i)
        #X = meanshift_step(X)   # slow implementation
        X = meanshift_step_batch(X)   # fast implementation
    return X

scale = 0.25    # downscale the image to run faster

# Load image and convert it to CIELAB space
image = rescale(io.imread('cow.jpg'), scale, multichannel=True) #image = rescale(io.imread('cow.jpg'), scale, multichannel=True)
image_lab = color.rgb2lab(image)
shape = image_lab.shape # record image shape
image_lab = image_lab.reshape([-1, 3])  # flatten the image 
# I guess the image is then an array of vector (3675, 3)

# Run your mean-shift algorithm
t = time.time()
#X = meanshift(torch.from_numpy(image_lab)).detach().cpu().numpy()

X = meanshift(torch.from_numpy(image_lab).cuda()).detach().cpu().numpy()  # you can use GPU if you have one
t = time.time() - t
print ('Elapsed time for mean-shift: {}'.format(t))

# Load label colors and draw labels as an image
colors = np.load('colors.npz')['colors']
colors[colors > 1.0] = 1
colors[colors < 0.0] = 0

centroids, labels = np.unique((X / 4).round(), return_inverse=True, axis=0)

assert np.max(labels) < len(colors), "labels values are not representable with the give colors"

result_image = colors[labels].reshape(shape)
result_image = rescale(result_image, 1 / scale, order=0, multichannel=True)     # resize result image to original resolution
result_image = (result_image * 255).astype(np.uint8)
io.imsave('result.png', result_image)
