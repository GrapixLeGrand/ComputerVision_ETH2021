import numpy as np
import os

"""
Load in a numpy array the 2D points that matched in 2 images.
The name (str) of both images must be contained in image_pair
"""
def ReadFeatureMatches(image_pair, data_folder):
  im1 = image_pair[0]
  im2 = image_pair[1]
  assert(im1 < im2)
  matchfile_path = os.path.join(data_folder, 'matches', im1 + '-' + im2 + '.txt')
  pair_matches = np.loadtxt(matchfile_path, dtype=int)
  return pair_matches

"""
Returns a numpy 2D array representing the K matrix
"""
def ReadKMatrix(data_folder):
  path = os.path.join(data_folder, 'images', 'K.txt')
  K = np.loadtxt(path)
  return K