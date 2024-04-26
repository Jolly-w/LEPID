import numpy as np
from scipy.spatial import distance


def classify(ex, model):
    k = 0
    clu_cent = []
    for k in range(len(model)):
        clu_cent.append(model[k].Mc_center)
    min_idx = find_nearest(ex, clu_cent)
    final_label = model[min_idx].Mc_label
    return final_label, min_idx

def find_nearest(arr, arr_list):
    min_dist = np.inf
    min_idx = -1
    for i, arr_elem in enumerate(arr_list):
        dist = np.linalg.norm(arr - arr_elem)
        if dist < min_dist:
            min_dist = dist
            min_idx = i
    return min_idx
