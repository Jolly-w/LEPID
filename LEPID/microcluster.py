import math as math
import numpy as np


class microcluster:
    """
    Implementation of the MicroCluster data structure for the CluStream algorithm
    Parameters
    ----------
    :parameter nb_points is the number of points in the cluster
    :parameter merge is used to indicate whether the cluster is resulting from the merge of two existing ones
    :parameter id_list is the id list of merged clusters
    :parameter linear_sum is the linear sum of the points in the cluster.
    :parameter squared_sum is  the squared sum of all the points added to the cluster.
    :parameter linear_time_sum is  the linear sum of all the timestamps of points added to the cluster.
    :parameter squared_time_sum is  the squared sum of all the timestamps of points added to the cluster.
    :parameter update_timestamp is used to indicate the last update time of the cluster.
    :parameter weight is the reliability of micro-cluster over time.
    :parameter Mc_labeling Mc_labeling indicates whether real labeling is applied to the current microcluster.
    :parameter W is importance of MCs
    :parameter Mc_special ：Wca
    """

    def __init__(self, nb_points=0, id_list=None, linear_sum=None, squared_sum=None,
                Mc_center=None, Mc_radius=None, update_timestamp=0, Mc_label=None, Mc_labeling=0,Mc_special=None,W=None):
        self.nb_points = nb_points
        self.id_list = id_list
        self.linear_sum = linear_sum
        self.squared_sum = squared_sum
        self.update_timestamp = update_timestamp
        self.Mc_center = Mc_center
        self.Mc_radius = Mc_radius
        self.Mc_label = Mc_label
        self.Mc_labeling = Mc_labeling
        self.Mc_special = Mc_special
        self.W = W


    def get_center(self):
        if self.nb_points == 0:
            return None
        center = [self.linear_sum[i] /
                self.nb_points for i in range(len(self.linear_sum))]
        return center

    def get_weight(self):
        return self.nb_points

    def insert(self, new_point, current_timestamp,Mc_labeling):
        self.nb_points += 1
        self.update_timestamp = current_timestamp
        for i in range(len(new_point)):
            self.linear_sum[i] += new_point[i]
            self.squared_sum[i] += math.pow(new_point[i], 2)
        self.Mc_center = [self.linear_sum[i] /
                  self.nb_points for i in range(len(self.linear_sum))]
        self.Mc_radius = np.sqrt(np.sum((self.squared_sum / self.nb_points) - (self.linear_sum / self.nb_points)**2))
        self.update_timestamp = current_timestamp
        if Mc_labeling:
            self.Mc_labeling += 1

    def merge(self, micro_cluster):
        # micro_cluster must be removed
        self.id_list = self.id_list
        self.Mc_radius = self.Mc_radius
        self.nb_points += micro_cluster.nb_points
        self.linear_sum += micro_cluster.linear_sum
        self.squared_sum += micro_cluster.squared_sum
        self.Mc_center += micro_cluster.Mc_center
        self.Mc_labeling += micro_cluster.Mc_labeling
        self.update_timestamp = max(self.update_timestamp, micro_cluster.update_timestamp)
        self.W = max(self.W, micro_cluster.W)
        #Mc_special
        self.Mc_special = max(self.Mc_special, micro_cluster.Mc_special)


    def get_variance_vec(self):
        variance_vec = list()
        for i in range(len(self.linear_sum)):
            ls_mean = self.linear_sum[i] / self.nb_points
            ss_mean = self.squared_sum[i] / self.nb_points
            variance = ss_mean - math.pow(ls_mean, 2)
            if variance <= 0:
                if variance > - self.epsilon:
                    variance = self.min_variance

            variance_vec.append(variance)
        return variance_vec

