from os import EX_CANTCREAT
import numpy as np
from numpy.lib.function_base import append

from impl.dlt import BuildProjectionConstraintMatrix
from impl.util import MakeHomogeneous, HNormalize
from impl.sfm.corrs import GetPairMatches
# from impl.opt import ImageResiduals, OptimizeProjectionMatrix

# # Debug
# import matplotlib.pyplot as plt
# from impl.vis import Plot3DPoints, PlotCamera, PlotProjectedPoints


def EstimateEssentialMatrix(K, im1, im2, matches):
  # TODO
  # Normalize coordinates (to points on the normalized image plane)
  # These are the keypoints on the normalized image plane (not to be confused with the normalization in the calibration exercise)
  
  #ones = np.ones(im1.kps.shape[0])
  #ones = ones[:, None]

  im1_pts = np.append(im1.kps, np.ones(im1.kps.shape[0])[:, None], axis=1)
  im2_pts = np.append(im2.kps, np.ones(im2.kps.shape[0])[:, None], axis=1)

  K_inv = np.linalg.inv(K)
  normalized_kps1 = (K_inv @ im1_pts.T).T
  normalized_kps2 = (K_inv @ im2_pts.T).T

  # TODO
  # Assemble constraint matrix
  constraint_matrix = np.zeros((matches.shape[0], 9))
  v = np.zeros(9)
  #v = v[:, None]

  # see https://en.wikipedia.org/wiki/Eight-point_algorithm
  for i in range(matches.shape[0]):
    # TODO
    # Add the constraints
    match_i = matches[i]
    x = normalized_kps1[match_i[0]]
    y = normalized_kps2[match_i[1]]

    v[0] = x[0] * y[0]
    v[1] = x[0] * y[1]
    v[2] = x[0]

    v[3] = x[1] * y[0]
    v[4] = x[1] * y[1]
    v[5] = x[1]

    v[6] = y[0]
    v[7] = y[1]
    v[8] = 1

    constraint_matrix[i] = v

  # Solve for the nullspace of the constraint matrix
  _, _, vh = np.linalg.svd(constraint_matrix)
  vectorized_E_hat = vh[-1,:]
  
  # TODO
  # Reshape the vectorized matrix to it's proper shape again
  E_hat = vectorized_E_hat.reshape((3, 3))

  # TODO
  # We need to fulfill the internal constraints of E
  # The first two singular values need to be equal, the third one zero.
  # Since E is up to scale, we can choose the two equal singluar values arbitrarily
  U, D, V = np.linalg.svd(E_hat)
  EI = np.eye(3)
  EI[0][0] = D[0]
  EI[1][1] = D[0]
  EI[2][2] = 0
  E = U @ EI @ V

  # This is just a quick test that should tell you if your estimated matrix is not correct
  # It might fail if you estimated E in the other direction (i.e. kp2' * E * kp1)
  # You can adapt it to your assumptions.
  for i in range(matches.shape[0]):
    kp1 = normalized_kps1[matches[i,0],:]
    kp2 = normalized_kps2[matches[i,1],:]

    print("VALS ",abs(kp1.transpose() @ E @ kp2))
    assert(abs(kp1.transpose() @ E @ kp2) < 0.01)

  return E


def DecomposeEssentialMatrix(E):

  u, s, vh = np.linalg.svd(E)

  # Determine the translation up to sign
  t_hat = u[:,-1]

  W = np.array([
    [0, -1, 0],
    [1, 0, 0],
    [0, 0, 1]
  ])

  # Compute the two possible rotations
  R1 = u @ W @ vh
  R2 = u @ W.transpose() @ vh

  # Make sure the orthogonal matrices are proper rotations
  if np.linalg.det(R1) < 0:
    R1 *= -1

  if np.linalg.det(R2) < 0:
    R2 *= -1

  # Assemble the four possible solutions
  sols = [
    (R1, t_hat),
    (R2, t_hat),
    (R1, -t_hat),
    (R2, -t_hat)
  ]

  return sols

def TriangulatePoints(K, im1, im2, matches):

  R1, t1 = im1.Pose()
  R2, t2 = im2.Pose()
  P1 = K @ np.append(R1, np.expand_dims(t1, 1), 1)
  P2 = K @ np.append(R2, np.expand_dims(t2, 1), 1)

  # Ignore matches that already have a triangulated point
  new_matches = np.zeros((0, 2), dtype=int)

  num_matches = matches.shape[0]
  for i in range(num_matches):
    p3d_idx1 = im1.GetPoint3DIdx(matches[i, 0])
    p3d_idx2 = im2.GetPoint3DIdx(matches[i, 1])
    if p3d_idx1 == -1 and p3d_idx2 == -1:
      new_matches = np.append(new_matches, matches[[i]], 0)

  #print(new_matches)
  #raise Exception("no")

  num_new_matches = new_matches.shape[0]

  points3D = np.zeros((num_new_matches, 3))

  for i in range(num_new_matches):

    kp1 = im1.kps[new_matches[i, 0], :]
    kp2 = im2.kps[new_matches[i, 1], :]

    """
    Here we solve Ax = 0 where A is constrained with the two matches and 
    Pi is the projection matrix of image i. Here x is the 3D point.
    """
    # H & Z Sec. 12.2
    A = np.array([
      kp1[0] * P1[2] - P1[0],
      kp1[1] * P1[2] - P1[1],
      kp2[0] * P2[2] - P2[0],
      kp2[1] * P2[2] - P2[1]
    ])

    _, _, vh = np.linalg.svd(A)
    homogeneous_point = vh[-1]
    #take all but the last elements and divide them by the last
    points3D[i] = homogeneous_point[:-1] / homogeneous_point[-1]


  # We need to keep track of the correspondences between image points and 3D points
  im1_corrs = new_matches[:,0]
  im2_corrs = new_matches[:,1]

  # TODO
  # Filter points behind the cameras by transforming them into each camera space and checking the depth (Z)
  # Make sure to also remove the corresponding rows in `im1_corrs` and `im2_corrs`

  #print(np.unique(points3D, axis=1))
  #raise Exception("no")

  idx1 = np.arange(points3D.shape[0])
  idx2 = np.arange(points3D.shape[0])

  # make 3D points homogenous
  points3DH = np.append(points3D, np.ones(points3D.shape[0])[:, None], axis=1)
  
  Cam1 = np.append(R1, np.expand_dims(t1, 1), 1)
  Cam2 = np.append(R2, np.expand_dims(t2, 1), 1)

  # project each 3D points on both images
  points3DCam1 = (P1 @ points3DH.T).T
  points3DCam2 = (P2 @ points3DH.T).T

  # extract the Z component of each
  Z1 = points3DCam1[:, 2]
  Z2 = points3DCam2[:, 2]

  # find the indices that fullfils a positive Z
  # on both images
  idx1 = idx1[Z1[idx1] >= 0]
  idx2 = idx2[Z2[idx2] >= 0]

  idx = np.intersect1d(idx1, idx2)

  #print(idx)
  points3D = points3D[idx]
  #print(points3D)
  #raise Exception("no")

  im1_corrs = im1_corrs[idx]
  im2_corrs = im2_corrs[idx]

  return points3D, im1_corrs, im2_corrs

def EstimateImagePose(points2D, points3D, K):  

  # TODO
  # We use points in the normalized image plane.
  # This removes the 'K' factor from the projection matrix.
  # We don't normalize the 3D points here to keep the code simpler.

  points2DH = np.append(points2D, np.ones(points2D.shape[0])[:, None], axis=1)

  K_inv = np.linalg.inv(K)
  normalized_points2D = (K_inv @ points2DH.T).T

  constraint_matrix = BuildProjectionConstraintMatrix(normalized_points2D, points3D)

  # We don't use optimization here since we would need to make sure to only optimize on the se(3) manifold
  # (the manifold of proper 3D poses). This is a bit too complicated right now.
  # Just DLT should give good enough results for this dataset.

  # Solve for the nullspace
  _, _, vh = np.linalg.svd(constraint_matrix)
  P_vec = vh[-1,:]
  P = np.reshape(P_vec, (3, 4), order='C')

  # Make sure we have a proper rotation
  u, s, vh = np.linalg.svd(P[:,:3])
  R = u @ vh

  if np.linalg.det(R) < 0:
    R *= -1

  _, _, vh = np.linalg.svd(P)
  C = np.copy(vh[-1,:])

  t = -R @ (C[:3] / C[3])

  return R, t

def TriangulateImage(K, image_name, images, registered_images, matches):

  # TODO 
  # Loop over all registered images and triangulate new points with the new image.
  # Make sure to keep track of all new 2D-3D correspondences, also for the registered images

  current_image = images[image_name] # get the current image
  points3D = np.zeros((0,3))
  #curr_image_2D_idx = np.zeros((0,))

  # You can save the correspondences for each image in a dict and refer to the `local` new point indices here.
  # Afterwards you just add the index offset before adding the correspondences to the images.
  corrs = {}

  for registered in registered_images:

    registered_image = images[registered]
    current_matches = GetPairMatches(image_name, registered, matches)
    points3D_fresh, im_corrs, r_corrs = TriangulatePoints(K, current_image, registered_image, current_matches)

    if (points3D_fresh.shape[0] > 0):
      
      corrs[registered] = (r_corrs, (points3D.shape[0], points3D.shape[0] + points3D_fresh.shape[0]))
      #curr_image_2D_idx = np.append(curr_image_2D_idx, im_corrs, axis=0)
      points3D = np.append(points3D, points3D_fresh, axis=0)
  
  # add the correspondance for the last image
  # corrs[image_name] = (curr_image_2D_idx, (0, points3D.shape[0]))

  return points3D, corrs

"""

if (image_name < registered):
      current_matches = matches[(image_name, registered)]
    else:
      current_matches = matches[(registered, image_name)]

if (image_name < registered):
  corrs[(image_name, registered)] = (im_corrs, r_corrs, (points3D.shape[0], points3D.shape[0] + points3D_fresh.shape[0]))
else:
  corrs[(registered, image_name)] = (r_corrs, im_corrs, (points3D.shape[0], points3D.shape[0] + points3D_fresh.shape[0]))
"""