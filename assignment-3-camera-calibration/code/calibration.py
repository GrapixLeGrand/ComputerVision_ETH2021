import matplotlib.pyplot as plt
import numpy as np

from impl.vis import Plot2DPoints, PlotProjectedPoints, Plot3DPoints, PlotCamera
from impl.calib.geometry import NormalizePoints2D, NormalizePoints3D, EstimateProjectionMatrix, DecomposeP
from impl.calib.io import ReadPoints2D, ReadPoints3D, ReadImageSize
from impl.calib.opt import ImageResiduals, OptimizeProjectionMatrix


def main():

  np.set_printoptions(precision=3)

  # Load the point correspondences
  # Each row is a point. Corresponding points are in the same rows in `points2D` and `points3D`, respectively.
  # 2D points as Nx2 array with [x, y] coordinates
  points2D = ReadPoints2D('../data')
  # 3D points as Nx3 array with [X, Y, Z] coordinates
  points3D = ReadPoints3D('../data')
  # Image size as [width, height]
  image_size = ReadImageSize('../data')

  # Visualize
  fig = plt.figure()
  ax3d = fig.add_subplot(121, projection='3d')
  ax2d = fig.add_subplot(122)
  Plot3DPoints(points3D, ax3d)
  Plot2DPoints(points2D, image_size, ax2d)
  plt.show(block=False)

  # TODO
  # Normalize 2D and 3D points

  """
  a) We do need data normalization because we need to find the fundamental matrix. And in order
  to find the fundamental matrix we need to solve a consequent linear system. Our data contains
  noise and therefore contains an amount of error. Thus, this amount of errors grow up as when
  performing the height points algorithm as the coefficients of the projected and backprojected 
  points will get mulitplied (see course slide). This will not affect the correctness of the height
  point algorithm as the system will still converge, it will just converge slower as opposed with
  normalized data.
  """

  
  normalized_points2D, T2D = NormalizePoints2D(points2D, image_size) # array of 2D row vectors
  normalized_points3D, T3D = NormalizePoints3D(points3D) # array of 3D row vectors
  
  """b) No answer for now page 88 of the book"""
  # TODO
  # Estimate the projection matrix from normalized correspondences
  P_hat = EstimateProjectionMatrix(normalized_points2D, normalized_points3D)

  # TODO
  # Optimize based on reprojection error
  P_hat_opt = OptimizeProjectionMatrix(P_hat, normalized_points2D, normalized_points3D)

  print(f'Reprojection error after optimization: {np.linalg.norm(ImageResiduals(P_hat_opt, normalized_points2D, normalized_points3D))**2}')

  # TODO
  # Denormalize P np.linalg.inv(T2D) 
  P = np.linalg.inv(T2D) @ P_hat_opt @ T3D

  # TODO
  # Decompose P
  K, R, t = DecomposeP(P)

  #raise NotImplementedError('distance function not implemented!')
  # Print the estimated values
  print(f'K=\n{K/K[2,2]}')
  print(f'R =\n{R}')
  print(f't = {t.transpose()}')

  # Visualize
  PlotCamera(R, t, ax3d, 0.5)
  PlotProjectedPoints(points3D, points2D, K, R, t, image_size, ax2d)

  # Make sure the plots are shown before the program terminates
  plt.show()

if __name__ == "__main__":
  main()