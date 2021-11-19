import numpy as np

"""
returns the indices of correspondance on the image to the other images. And also
from the 2D image to the 3D points
"""
# Find (unique) 2D-3D correspondences from 2D-2D correspondences
def Find2D3DCorrespondences(image_name, images, matches, registered_images):
  assert(image_name not in registered_images)

  image_kp_idxs = []
  p3D_idxs = []
  for other_image_name in registered_images:
    other_image = images[other_image_name]
    pair_matches = GetPairMatches(image_name, other_image_name, matches)

    for i in range(pair_matches.shape[0]):
      p3D_idx = other_image.GetPoint3DIdx(pair_matches[i,1])
      if p3D_idx > -1:
        p3D_idxs.append(p3D_idx)
        image_kp_idxs.append(pair_matches[i,0])

  print(f'found {len(p3D_idxs)} points, {np.unique(np.array(p3D_idxs)).shape[0]} unique points')

  # Remove duplicated correspondences
  _, unique_idxs = np.unique(np.array(p3D_idxs), return_index=True)
  image_kp_idxs = np.array(image_kp_idxs)[unique_idxs].tolist()
  p3D_idxs = np.array(p3D_idxs)[unique_idxs].tolist()
  
  return image_kp_idxs, p3D_idxs


# Make sure we get keypoint matches between the images in the order that we requested
def GetPairMatches(im1, im2, matches):
  if im1 < im2:
    return matches[(im1, im2)]
  else:
    return np.flip(matches[(im2, im1)], 1)

# Update the reconstruction with the new information from a triangulated image
def UpdateReconstructionState(new_points3D, corrs, points3D, images):

  # e_im1.Add3DCorrs(im1_corrs, list(range(points3D.shape[0])))
  
  # TODO
  # Add the new points to the set of reconstruction points and add the correspondences to the images.
  # Be careful to update the point indices to the global indices in the `points3D` array.
  offset = points3D.shape[0]
  points3D = np.append(points3D, new_points3D, 0)

  for im_name in corrs:
    corr = corrs[im_name][0]
    l, u = corrs[im_name][1]
    images[im_name].Add3DCorrs(corr, np.arange(l, u) + offset)
    
  return points3D, images

"""
    img_1 = im_name[0]
    img_2 = im_name[1]

    corr_1 = corrs[im_name][0]
    corr_2 = corrs[im_name][1]

    pts_fresh = corrs[im_name][2]
    _, all_new_idx, current_new_idx = np.intersect1d(new_points3D, pts_fresh, return_indices=True)

    print(all_new_idx, " new ", all_new_idx.shape[0])
    print(current_new_idx, " cur ", current_new_idx.shape[0])
  
    images[img_1].Add3DCorrs(corr_1, current_new_idx + offset)
    images[img_2].Add3DCorrs(corr_2, current_new_idx + offset)
"""