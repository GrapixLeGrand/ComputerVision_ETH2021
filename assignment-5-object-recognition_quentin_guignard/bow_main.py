import numpy as np
import cv2
import glob
import os
from sklearn.cluster import KMeans
from tqdm import tqdm


def findnn(D1, D2):
    """
    :param D1: NxD matrix containing N feature vectors of dim. D
    :param D2: MxD matrix containing M feature vectors of dim. D
    :return:
        Idx: N-dim. vector containing for each feature vector in D1 the index of the closest feature vector in D2.
        Dist: N-dim. vector containing for each feature vector in D1 the distance to the closest feature vector in D2
    """
    N = D1.shape[0]
    M = D2.shape[0]  # [k]

    # Find for each feature vector in D1 the nearest neighbor in D2
    Idx, Dist = [], []
    for i in range(N):
        minidx = 0
        mindist = np.linalg.norm(D1[i, :] - D2[0, :])
        for j in range(1, M):
            d = np.linalg.norm(D1[i, :] - D2[j, :])

            if d < mindist:
                mindist = d
                minidx = j
        Idx.append(minidx)
        Dist.append(mindist)
    return Idx, Dist


def grid_points(img, nPointsX, nPointsY, border):
    """
    :param img: input gray img, numpy array, [h, w]
    :param nPointsX: number of grids in x dimension
    :param nPointsY: number of grids in y dimension
    :param border: leave border pixels in each image dimension
    :return: vPoints: 2D grid point coordinates, numpy array, [nPointsX*nPointsY, 2]
    """

    #TODO  ex 2.1

    H, W = img.shape

    start = border - 1
    end_W = W - border - 1
    end_H = H - border - 1

    points_W = np.linspace(start, end_W, num=nPointsX, dtype=int)
    points_H = np.linspace(start, end_H, num=nPointsY, dtype=int)
    
    xW, xH = np.meshgrid(points_W, points_H)

    xW = xW.flatten()
    xH = xH.flatten()

    vPoints = np.stack((xW, xH), axis=1) # numpy array, [nPointsX*nPointsY, 2]

    return vPoints



def descriptors_hog(img, vPoints, cellWidth, cellHeight):
    nBins = 8

    # cv2.CV_16S
    grad_x = cv2.Sobel(img, cv2.CV_32F, dx=1, dy=0, ksize=1)
    grad_y = cv2.Sobel(img, cv2.CV_32F, dx=0, dy=1, ksize=1)

    angles = (np.arctan2(grad_x, grad_y) * 180 / np.pi) + 180

    descriptors = np.zeros((len(vPoints), 128)) #[]  # list of descriptors for the current image, each entry is one 128-d vector for a grid point
    for i in range(len(vPoints)):

        #TODO
        x = vPoints[i][0] # 2D point
        y = vPoints[i][1]

        start_W, end_W = x - (2 * cellWidth - 1), x + 2 * cellWidth + 1
        start_H, end_H = y - (2 * cellHeight - 1), y + 2 * cellHeight + 1

        gp_grad_angles = angles[start_H:end_H, start_W:end_W]
        
        current_descriptor = np.zeros(128)

        # compute for each cell a descriptor and concatenate them (16 cells)
        for yy in range(0, 4):
            for xx in range(0, 4):
                
                # get the subset conserning some cell
                start_cell_W = cellWidth * xx 
                end_cell_W = cellWidth * (xx + 1) 
                start_cell_H = cellHeight * yy
                end_cell_H = cellHeight * (yy + 1)

                cell_grad_angles = gp_grad_angles[start_cell_W:end_cell_W, start_cell_H:end_cell_H]
                
                current_cell_descriptor = np.zeros(8)

                # fill the bins with the current values
                for x_cell in range(0, cellWidth):
                    for y_cell in range(0, cellHeight):
                        
                        grad_angle = cell_grad_angles[x_cell][y_cell]

                        bin = int((grad_angle / 360.0) * (nBins - 1))
                        current_cell_descriptor[bin] += 1

                start_index = xx * 8 + yy * 32
                end_index = (xx + 1) * 8 + yy * 32
                current_descriptor[start_index:end_index] = current_cell_descriptor

        descriptors[i] = current_descriptor

    return descriptors


def create_codebook(nameDirPos, nameDirNeg, k, numiter):
    """
    :param nameDirPos: dir to positive training images
    :param nameDirNeg: dir to negative training images
    :param k: number of kmeans cluster centers
    :param numiter: maximum iteration numbers for kmeans clustering
    :return: vCenters: center of kmeans clusters, numpy array, [k, 128]
    """
    vImgNames = sorted(glob.glob(os.path.join(nameDirPos, '*.png')))
    vImgNames = vImgNames + sorted(glob.glob(os.path.join(nameDirNeg, '*.png')))

    nImgs = len(vImgNames)

    cellWidth = 4
    cellHeight = 4
    nPointsX = 10
    nPointsY = 10
    border = 8

    vFeatures = []  # list for all features of all images (each feature: 128-d, 16 histograms containing 8 bins)
    # Extract features for all image
    for i in tqdm(range(nImgs)):
        print('processing image {} ...'.format(i+1))
        img = cv2.imread(vImgNames[i])  # [172, 208, 3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # [h, w]

        # Collect local feature points for each image, and compute a descriptor for each local feature point
        # TODO
        vPoints = grid_points(img, nPointsX, nPointsY, border)
        vFeatures.append(descriptors_hog(img, vPoints, cellWidth, cellHeight))

    vFeatures = np.asarray(vFeatures)  # [n_imgs, n_vPoints, 128]
    vFeatures = vFeatures.reshape(-1, vFeatures.shape[-1])  # [n_imgs*n_vPoints, 128]
    print('number of extracted features: ', len(vFeatures))


    # Cluster the features using K-Means
    print('clustering ...')
    kmeans_res = KMeans(n_clusters=k, max_iter=numiter).fit(vFeatures)
    vCenters = kmeans_res.cluster_centers_  # [k, 128]
    return vCenters


def bow_histogram(vFeatures, vCenters):
    """
    :param vFeatures: MxD matrix containing M feature vectors of dim. D
    :param vCenters: NxD matrix containing N cluster centers of dim. D
    :return: histo: N-dim. numpy vector containing the resulting BoW activation histogram.
    """
    #histo = None

    # TODO
    M, D = vFeatures.shape
    N, _ = vCenters.shape # 10 clusters each 128-D

    squared_distances = vFeatures @ vCenters.T
    min_dists = np.argmin(squared_distances, axis=1)

    histo = np.zeros(N)

    elements, counts = np.unique(min_dists, return_counts=1)

    for i in range(0, len(elements)):
        e = elements[i]
        count = counts[i]
        histo[e] += count

    return histo


def create_bow_histograms(nameDir, vCenters):
    """
    :param nameDir: dir of input images
    :param vCenters: kmeans cluster centers, [k, 128] (k is the number of cluster centers)
    :return: vBoW: matrix, [n_imgs, k]
    """
    vImgNames = sorted(glob.glob(os.path.join(nameDir, '*.png')))
    nImgs = len(vImgNames)

    cellWidth = 4
    cellHeight = 4
    nPointsX = 10
    nPointsY = 10
    border = 8

    # Extract features for all images in the given directory
    K, _ = vCenters.shape
    vBoW = []
    for i in tqdm(range(nImgs)):
        print('bow processing image {} ...'.format(i + 1))
        img = cv2.imread(vImgNames[i])  # [172, 208, 3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # [h, w]

        # TODO

        vPoints = grid_points(img, nPointsX, nPointsY, border)
        vFeatures = descriptors_hog(img, vPoints, cellWidth, cellHeight)
        
        vBoW.append(bow_histogram(vFeatures, vCenters))


    vBoW = np.asarray(vBoW)  # [n_imgs, k]
    return vBoW



def bow_recognition_nearest(histogram,vBoWPos,vBoWNeg):
    """
    :param histogram: bag-of-words histogram of a test image, [1, k]
    :param vBoWPos: bag-of-words histograms of positive training images, [n_imgs, k]
    :param vBoWNeg: bag-of-words histograms of negative training images, [n_imgs, k]
    :return: sLabel: predicted result of the test image, 0(without car)/1(with car)
    """

    DistPos, DistNeg = None, None

    # Find the nearest neighbor in the positive and negative sets and decide based on this neighbor
    # todo
    dist_to_all_pos = np.diag((histogram - vBoWPos).dot((histogram - vBoWPos).T))
    dist_to_all_neg = np.diag((histogram - vBoWNeg).dot((histogram - vBoWNeg).T))

    min_index_pos = np.argmin(dist_to_all_pos)
    min_index_neg = np.argmin(dist_to_all_neg)
    
    DistPos = dist_to_all_pos[min_index_pos]
    DistNeg = dist_to_all_neg[min_index_neg]

    if (DistPos < DistNeg):
        sLabel = 1
    else:
        sLabel = 0
    return sLabel


if __name__ == '__main__':
    nameDirPos_train = 'data/data_bow/cars-training-pos'
    nameDirNeg_train = 'data/data_bow/cars-training-neg'
    nameDirPos_test = 'data/data_bow/cars-testing-pos'
    nameDirNeg_test = 'data/data_bow/cars-testing-neg'


    k = 17 #None  # todo
    numiter = 10000 #None  # todo

    print('creating codebook ...')
    vCenters = create_codebook(nameDirPos_train, nameDirNeg_train, k, numiter)

    print('creating bow histograms (pos) ...')
    vBoWPos = create_bow_histograms(nameDirPos_train, vCenters)
    print('creating bow histograms (neg) ...')
    vBoWNeg = create_bow_histograms(nameDirNeg_train, vCenters)

    # test pos samples
    print('creating bow histograms for test set (pos) ...')
    vBoWPos_test = create_bow_histograms(nameDirPos_test, vCenters)  # [n_imgs, k]
    result_pos = 0
    print('testing pos samples ...')
    for i in range(vBoWPos_test.shape[0]):
        cur_label = bow_recognition_nearest(vBoWPos_test[i:(i+1)], vBoWPos, vBoWNeg)
        result_pos = result_pos + cur_label
    acc_pos = result_pos / vBoWPos_test.shape[0]
    print('test pos sample accuracy:', acc_pos)

    # test neg samples
    print('creating bow histograms for test set (neg) ...')
    vBoWNeg_test = create_bow_histograms(nameDirNeg_test, vCenters)  # [n_imgs, k]
    result_neg = 0
    print('testing neg samples ...')
    for i in range(vBoWNeg_test.shape[0]):
        cur_label = bow_recognition_nearest(vBoWNeg_test[i:(i + 1)], vBoWPos, vBoWNeg)
        result_neg = result_neg + cur_label
    acc_neg = 1 - result_neg / vBoWNeg_test.shape[0]
    print('test neg sample accuracy:', acc_neg)
