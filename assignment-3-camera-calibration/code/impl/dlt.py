import numpy as np

def BuildProjectionConstraintMatrix(points2D, points3D):

  # TODO
  # For each correspondence, build the two rows of the constraint matrix and stack them
  num_corrs = points2D.shape[0]
  constraint_matrix = np.zeros((num_corrs * 2, 12))

  #The input are 3D and 2D points
  zero = np.zeros(points3D.shape[1] + 1) # same shape as X 4D vectors of zeros

  # Since P is of the form 3x4, then the P vector that we need to find is 
  # a 12D vector and so the constraints should be a set of row vectors of
  # length 12 too.
  for i in range(num_corrs):
    # TODO Add your code here
    X = np.append(points3D[i], 1) # 4D vectors (I think)
    x = points2D[i] # 2D vector

    constraint_matrix[2 * i + 0] = np.hstack((zero, -X, x[1] * X))
    constraint_matrix[2 * i + 1] = np.hstack((X, zero, -x[0] * X))

  return constraint_matrix