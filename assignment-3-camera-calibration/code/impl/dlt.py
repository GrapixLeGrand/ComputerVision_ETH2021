import numpy as np

def BuildProjectionConstraintMatrix(points2D, points3D):

  # TODO
  # For each correspondence, build the two rows of the constraint matrix and stack them

  num_corrs = points2D.shape[0]
  constraint_matrix = np.zeros((num_corrs * 2, 12))

  zero = np.zeros(points3D.shape[1] + 1) # same shape as X 4D vectors of zeros

  for i in range(num_corrs):
    # TODO Add your code here
    X = np.append(points3D[i], 1) # 4D vectors (I think)
    x = points2D[i] # 3D vector ?

    #print("constrain matrix: ", constraint_matrix)
    #print("one vector homo: ", np.append(X, 1))
    #print("3D points: ", points3D)
    #print("2D points: ", points2D)
    constraint_matrix[2 * i + 0] = np.hstack((zero, -X, x[1] * X))
    constraint_matrix[2 * i + 1] = np.hstack((X, zero, -x[0] * X))

  return constraint_matrix