from sklearn.metrics.pairwise import pairwise_distances
from scipy.optimize import linear_sum_assignment
import numpy as np


# calculate jaccard and RMSE given two arrays of xyz's and the radius for matching
# matching is done based on the hungarian algorithm, where all coords. are given in microns
def calc_jaccard_index(xyz_gt, xyz_rec, radius):
    
    # if the net didn't detect anything return None's
    if xyz_rec is None or len(xyz_rec) == 0:
        print("Empty Prediction!")
        return 0.0
    
    if xyz_gt is None or len(xyz_gt) == 0:
        print("Empty Ground Truth!")
        return 0.0
    
    else:
        # make sure the inputs are 2D arrays
        xyz_gt = np.atleast_2d(xyz_gt)
        xyz_rec = np.atleast_2d(xyz_rec)
        
        # calculate the distance matrix for each GT to each prediction
        C = pairwise_distances(xyz_rec, xyz_gt, 'euclidean')
        
        # number of recovered points and GT sources
        num_rec = xyz_rec.shape[0]
        num_gt = xyz_gt.shape[0]
        
        # find the matching using the Hungarian algorithm
        rec_ind, gt_ind = linear_sum_assignment(C)
        
        # number of matched points
        num_matches = len(rec_ind)
        
        # run over matched points and filter points radius away from GT
        indicatorTP = [False]*num_matches
        for i in range(num_matches):
            
            # if the point is closer than radius then TP else it's FP
            if C[rec_ind[i], gt_ind[i]] < radius:
                indicatorTP[i] = True
        
        # resulting TP count
        TP = sum(indicatorTP)

        # resulting jaccard index
        return TP / (num_rec + num_gt - TP)