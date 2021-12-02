import numpy as np
from matplotlib import pyplot as plt
import random

# to install open3d : sudo pip3 install open3d==0.13.0

np.random.seed(0)
random.seed(0)


"""
Result:

Estimated coefficients (true, linear regression, RANSAC):
1 10 0.6159656578755459 8.96172714144364 1.0393762271934441 9.950760272211799
"""

def least_square(x,y):
	# TODO
	# return the least-squares solution
	# you can use np.linalg.lstsq

	# build the constraint matrix
	A = np.vstack([x, np.ones(len(x))]).T
	k, b = np.linalg.lstsq(A, y, rcond=None)[0]

	return k, b

def num_inlier(x,y,k,b,n_samples,thres_dist):
	# TODO
	# compute the number of inliers and a mask that denotes the indices of inliers

	dists = np.abs(y - k * x - b) / np.sqrt(1 + k**2)
	current_inlier_mask = dists < thres_dist
	current_inliner_num = len(np.where(dists < thres_dist))

	return current_inliner_num, current_inlier_mask

def ransac(x,y,iter,n_samples,thres_dist,num_subset):
	# TODO
	# ransac

	k_ransac = None
	b_ransac = None
	inlier_mask = None
	best_inliers = 0
	points_indices = list(range(n_samples))

	i = 0
	while i < iter:

		# 1: randomly choose a small subset
		#samples_indices = np.random.choice(num_subset, num_subset, replace=False)
		samples_indices = np.array(random.sample(points_indices, num_subset))
		rest_indices = np.setdiff1d(np.arange(n_samples), samples_indices)

		x_samples = x[samples_indices]
		y_samples = y[samples_indices]

		#x_rest = x[rest_indices]
		#y_rest = y[rest_indices]

		# 2 : compute the least-squares solution for this subset

		k_ransac, b_ransac = least_square(x_samples, y_samples)

		# 3 : compute the number of inliers and the mask denotes the indices ofinliers
		
		# we need to compute the distance of the rest to the line
		current_inliner_num, current_inlier_mask = num_inlier(x,y,k_ransac,b_ransac,n_samples,thres_dist)

		# 4 : update best result if num_inliner is more than best
		if (current_inliner_num > best_inliers):
			best_inliers = len(current_inlier_mask)
			inlier_mask = current_inlier_mask


		i += 1

	k_ransac, b_ransac = least_square(x[inlier_mask], y[inlier_mask])

	return k_ransac, b_ransac, inlier_mask

def main():
	iter = 300
	thres_dist = 1
	n_samples = 500
	n_outliers = 50
	k_gt = 1
	b_gt = 10
	num_subset = 5
	x_gt = np.linspace(-10,10,n_samples)
	print(x_gt.shape)
	y_gt = k_gt*x_gt+b_gt
	# add noise
	x_noisy = x_gt+np.random.random(x_gt.shape)-0.5
	y_noisy = y_gt+np.random.random(y_gt.shape)-0.5
	# add outlier
	x_noisy[:n_outliers] = 8 + 10 * (np.random.random(n_outliers)-0.5)
	y_noisy[:n_outliers] = 1 + 2 * (np.random.random(n_outliers)-0.5)

	# least square
	k_ls, b_ls = least_square(x_noisy, y_noisy)

	# ransac
	k_ransac, b_ransac, inlier_mask = ransac(x_noisy, y_noisy, iter, n_samples, thres_dist, num_subset)
	outlier_mask = np.logical_not(inlier_mask)

	print("Estimated coefficients (true, linear regression, RANSAC):")
	print(k_gt, b_gt, k_ls, b_ls, k_ransac, b_ransac)

	line_x = np.arange(x_noisy.min(), x_noisy.max())
	line_y_ls = k_ls*line_x+b_ls
	line_y_ransac = k_ransac*line_x+b_ransac

	plt.scatter(
	    x_noisy[inlier_mask], y_noisy[inlier_mask], color="yellowgreen", marker=".", label="Inliers"
	)
	plt.scatter(
	    x_noisy[outlier_mask], y_noisy[outlier_mask], color="gold", marker=".", label="Outliers"
	)
	plt.plot(line_x, line_y_ls, color="navy", linewidth=2, label="Linear regressor")
	plt.plot(
	    line_x,
	    line_y_ransac,
	    color="cornflowerblue",
	    linewidth=2,
	    label="RANSAC regressor",
	)
	plt.legend()
	plt.xlabel("Input")
	plt.ylabel("Response")
	plt.show()

if __name__ == '__main__':
	main()