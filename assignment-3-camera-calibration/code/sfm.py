import gc
import itertools
import matplotlib.pyplot as plt
import numpy as np

from impl.vis import Plot3DPoints
from impl.sfm.corrs import Find2D3DCorrespondences, GetPairMatches, UpdateReconstructionState
from impl.sfm.geometry import DecomposeEssentialMatrix, EstimateEssentialMatrix, TriangulatePoints, TriangulateImage, EstimateImagePose
from impl.sfm.image import Image
from impl.sfm.io import ReadFeatureMatches, ReadKMatrix
from impl.sfm.vis import PlotImages, PlotWithKeypoints, PlotImagePairMatches, PlotCameras

"""
def points_in_front(im1, im2, K, R, t, matches):

  x_pts = im1.kps #np.append(im1, np.ones(im1.kps.shape[0])[:, None], axis=1)
  y_pts = im2.kps #np.append(im2.kps, np.ones(im2.kps.shape[0])[:, None], axis=1)

  im1_pts = np.append(im1.kps, np.ones(im1.kps.shape[0])[:, None], axis=1)
  im2_pts = np.append(im2.kps, np.ones(im2.kps.shape[0])[:, None], axis=1)

  K_inv = np.linalg.inv(K)
  x_pts = (K_inv @ im1_pts.T).T
  y_pts = (K_inv @ im2_pts.T).T

  P_x = np.append(R, np.expand_dims(t, 1), 1)
  P_y = np.append(np.eye(3), np.expand_dims(np.zeros(3),1), 1)

  A = np.zeros((4, 4))

  X_pts = np.zeros((matches.shape[0], 3)) # the triangulated points

  #set points
  for i in range(matches.shape[0]):

    match_i = matches[i]
    x = x_pts[match_i[0]]
    y = y_pts[match_i[1]]

    A[0] = x[0] * P_x[2] - P_x[0]
    A[1] = x[1] * P_x[2] - P_x[1]
    
    A[2] = y[0] * P_y[2] - P_y[0]
    A[3] = y[1] * P_y[2] - P_y[1]

    _, _, vh = np.linalg.svd(A)
    X_tri = vh[-1,:]
    X_tri /= X_tri[3]

    X_pts[i] = X_tri[:3]

  X_z = X_pts[:, 2]
  behind = np.count_nonzero(np.where(X_z < 0))

  return X_pts, behind
"""

def main():

  np.set_printoptions(linewidth=10000, edgeitems=100, precision=3)

  data_folder = '../data'
  image_names = [
    '0000.png',
    '0001.png',
    '0002.png',
    '0003.png',
    '0004.png',
    '0005.png',
    '0006.png',
    '0007.png',
    '0008.png',
    '0009.png']

  # Read images
  images = {}
  for im_name in image_names:
    images[im_name] = (Image(data_folder, im_name))

  # Read the matches
  matches = {}
  for image_pair in itertools.combinations(image_names, 2):
    matches[image_pair] = ReadFeatureMatches(image_pair, data_folder)

  K = ReadKMatrix(data_folder)

  """
  We select arbitrarily the images that we will register to start the algorithm.
  """
  init_images = [3, 4]

  # Visualize images and features
  # You can comment these lines once you verified that the images are loaded correctly

  # Show the images
  #PlotImages(images)

  # Show the keypoints
  for image_name in image_names:
    #PlotWithKeypoints(images[image_name])
    None 

  # Show the feature matches
  for image_pair in itertools.combinations(image_names, 2):
    #PlotImagePairMatches(images[image_pair[0]], images[image_pair[1]], matches[(image_pair[0], image_pair[1])])
    gc.collect()
    None
  
  e_im1_name = image_names[init_images[0]]
  e_im2_name = image_names[init_images[1]]
  e_im1 = images[e_im1_name]
  e_im2 = images[e_im2_name]
  e_matches = GetPairMatches(e_im1_name, e_im2_name, matches)

  #print("e_im1: ", e_im1_name)
  #print("k: ",K)
  #print(e_im1.kps)

  # TODO Estimate relative pose of first pair
  # Estimate Fundamental matrix
  E = EstimateEssentialMatrix(K, e_im1, e_im2, e_matches)

  # Extract the relative pose from the essential matrix.
  # This gives four possible solutions and we need to check which one is the correct one in the next step
  possible_relative_poses = DecomposeEssentialMatrix(E)

  """
  E is normalized correspondance mapping so x_hat = [R|t]X
  """

  # TODO
  # For each possible relative pose, try to triangulate points.
  # We can assume that the correct solution is the one that gives the most points in front of both cameras
  # Be careful not to set the transformation in the wrong direction
  
  # TODO
  # Set the image poses in the images (image.SetPose(...))
  points3D  = None
  im1_corrs = None
  im2_corrs = None
  is_first  = True
  min_behind = 0

  for R, t in possible_relative_poses:
    e_im1.SetPose(R, t)
    e_im2.SetPose(np.eye(3), np.zeros(t.shape))
    try3D, try_im1_corrs, try_im2_corrs = TriangulatePoints(K, e_im1, e_im2, e_matches)
    behind = try3D.shape[0]
    #print("b: ", behind)
    if (is_first or behind > min_behind):
      #print("pts: ", points3D)
      points3D = try3D
      im1_corrs = try_im1_corrs
      im2_corrs = try_im2_corrs
      min_behind = behind
      is_first = False
  
  assert points3D is not None, "points None"
  assert points3D.shape[0] != 0, "empty 3Dpoints !"
  assert im1_corrs is not None, "correlations 1 None"
  assert im1_corrs is not None, "correlations 2 None"
  
  #print(points3D)
  #raise Exception("no")
  
  # TODO Triangulate initial points
  #points3D, im1_corrs, im2_corrs = TriangulatePoints(K, e_im1, e_im2, e_matches)

  # Add the new 2D-3D correspondences to the images
  e_im1.Add3DCorrs(im1_corrs, list(range(points3D.shape[0])))
  e_im2.Add3DCorrs(im2_corrs, list(range(points3D.shape[0])))

  # Keep track of all registered images
  registered_images = [e_im1_name, e_im2_name]

  for reg_im in registered_images:
    print(f'Image {reg_im} sees {images[reg_im].NumObserved()} 3D points')

  #raise Exception("no")

  """
  We will triangulate all images successively. Each image is registered if we can
  find at least 50 points on it that would belong to other images too.

  We are trying to reconstruct the positions of the camera in 3D from the sequence of image.
  The algorithm registered first two images (selected arbitrarily). We then extracted 3D points
  from those two first images by finding the correspondance matrix E and by triangulating all the
  3D points. We know want to register all the other images. This means that for a non-registered
  2D image we want to find some correspondances (at least 50 points) as in those that matched 
  until know. We will then do as in the first part, try to estimate the pose and validate it
  by triangulating the points. We are done once we could do it for all images.
  """
  # Register new images + triangulate
  # Run until we can register all images
  while len(registered_images) < len(images):
    for image_name in images:
      
      if image_name in registered_images:
        continue
      
      """
      Given the matches that we have found in the two first images of the first part
      We want to find the 2D correspondance on the current image if there are ones.
      This process will iterate over all images 
      """
      # Find 2D-3D correspondences
      image_kp_idxs, point3D_idxs = Find2D3DCorrespondences(image_name, images, matches, registered_images)

      """
      If we could find enought corresondances we will register the image
      """
      # With two few correspondences the pose estimation becomes shaky.
      # Keep this image for later
      if len(image_kp_idxs) < 50:
        continue

      print(f'Register image {image_name} from {len(image_kp_idxs)} correspondences')

      """
      use the 2D points that we could find on other images too. Basically, we filter the 
      original 2D point of the image to be those matching our correspondances with the actual
      registered images.
      """
      # Estimate new image pose
      R, t = EstimateImagePose(images[image_name].kps[image_kp_idxs], points3D[point3D_idxs], K)

      # Set the estimated image pose in the image and add the correspondences between keypoints and 3D points
      images[image_name].SetPose(R, t)
      images[image_name].Add3DCorrs(image_kp_idxs, point3D_idxs)

      # TODO
      # Triangulate new points wth all previously registered images
      image_points3D, corrs = TriangulateImage(K, image_name, images, registered_images, matches)

      # TODO
      # Update the 3D points and image correspondences
      points3D, images = UpdateReconstructionState(image_points3D, corrs, points3D, images)

      registered_images.append(image_name)


  # Visualize
  fig = plt.figure()
  ax3d = fig.add_subplot(111, projection='3d')
  Plot3DPoints(points3D, ax3d)
  PlotCameras(images, registered_images, ax3d)

  # Delay termination of the program until the figures are closed
  # Otherwise all figure windows will be killed with the program
  plt.show(block=True)


if __name__ == '__main__':
  main()