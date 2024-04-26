from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_array
from sklearn.cluster import KMeans
from microcluster import microcluster as model
from scipy.spatial import distance
import math
import numpy as np
import threading
import time
from config import *
from utils import *
from math import log
import sys



class CluStream(BaseEstimator, ClusterMixin):

    def __init__(self, nb_initial_points=100, time_window=1000, nb_micro_cluster=50,
                 nb_macro_cluster=5, micro_clusters=[], timestamp=0):
        self.start_time = 0
        self.nb_initial_points = nb_initial_points
        self.time_window = time_window  # Range of the window
        self.micro_clusters = micro_clusters
        self.nb_micro_cluster = nb_micro_cluster
        self.nb_macro_cluster = nb_macro_cluster
        self.nb_created_clusters = 0

    # def fit(self, X, Y=None):
    def fit(self, X, label_list):
        X = check_array(X, accept_sparse='csr')
        nb_initial_points = X.shape[0]
        if nb_initial_points >= self.nb_initial_points:
            kmeans = KMeans(n_clusters=self.nb_micro_cluster, random_state=1)
            micro_cluster_labels = kmeans.fit_predict(X)
            X = np.column_stack((micro_cluster_labels, X))
            initial_clusters = [X[X[:, 0] == l][:, 1:] for l in set(micro_cluster_labels) if l != -1]
            for cluster in initial_clusters:
                self.create_micro_cluster(cluster, label_list)
        self.start_time = 0
    def create_micro_cluster(self, cluster,label_list):
        Mc_label = label_list
        nb_points = cluster.shape[0]
        linear_sum = np.sum(cluster, axis=0)
        squared_sum = np.sum(np.square(cluster), axis=0)
        Mc_center = linear_sum/nb_points
        Mc_radius = math.sqrt(np.sum(squared_sum/nb_points) - np.sum(np.square(linear_sum/nb_points)))
        #
        new_m_cluster = model(nb_points=nb_points, linear_sum=linear_sum, squared_sum=squared_sum,
                              Mc_center=Mc_center, Mc_radius=Mc_radius, Mc_label=Mc_label, update_timestamp=0,Mc_labeling=1,Mc_special=0,W=1)
        self.micro_clusters.append(new_m_cluster)

    def partial_fit(self, ex, CurrentTime):
        self.timestamp = CurrentTime
        x = ex['data']
        closest_cluster, p_id = self.find_closest_cluster(x, self.micro_clusters)
        p_label = closest_cluster.Mc_label
        return p_label, p_id

    def find_closest_cluster(self, x, micro_clusters):
        min_distance = sys.float_info.max
        for cluster in micro_clusters:
            distance_cluster = self.distance_to_cluster(x, cluster)
            if distance_cluster < min_distance:
                min_distance = distance_cluster
                closest_cluster = cluster
                cluster_id = micro_clusters.index(cluster)
        return closest_cluster, cluster_id

    def distance_to_cluster(self, x, cluster):
        cluster_Mc_center = cluster.Mc_center
        # if len(cluster.Mc_center) > 8:
        #     cluster_Mc_center = cluster_Mc_center[:8]
        return distance.euclidean(x, cluster_Mc_center)

    def update_cluster(self, x, cluster, CurrentTime,labeling):
        cluster.insert(new_point=x,current_timestamp=CurrentTime,Mc_labeling=labeling)

    def creat(self,x,y,CurrentTime,Mc_labeling,Mc_spe):
        if len(self.micro_clusters) >= maxMcs:
            self.merge_cluster()
        X = x
        linear_sum = X
        squared_sum = np.square(X)
        Mc_center = X
        closestMc,MC_number = self.find_closest_cluster(X,self.micro_clusters)
        Mc_radius = self.micro_clusters[MC_number].Mc_radius
        update_timestamp = CurrentTime
        Mc_label = y
        Mc_labeling = Mc_labeling
        Mc_special = Mc_spe
        new_m_cluster = model(nb_points=1,linear_sum=linear_sum, squared_sum=squared_sum, update_timestamp=update_timestamp,
                              Mc_center=Mc_center,Mc_radius=Mc_radius,Mc_label=Mc_label,Mc_labeling=Mc_labeling,Mc_special=Mc_special,W=1)
        self.micro_clusters.append(new_m_cluster)

    def creat_new(self,X,y,cluster,Ra,CurrentTime):
        np = int(cluster.nb_points * Ra)
        if np == 0:
            nb_points = 1
        else:
            nb_points = np
        linear_sum = cluster.linear_sum*Ra
        squared_sum = cluster.squared_sum*Ra*Ra
        Mc_center = X*Ra
        Mc_radius = cluster.Mc_radius*Ra
        update_timestamp = CurrentTime
        Mc_label = y
        new_m_cluster = model(nb_points=nb_points,linear_sum=linear_sum, squared_sum=squared_sum, update_timestamp=update_timestamp,
                              Mc_center=Mc_center,Mc_radius=Mc_radius,Mc_label=Mc_label,Mc_labeling=1,Mc_special=2,W=1)
        self.micro_clusters.append(new_m_cluster)


    def merge_cluster(self):
        min_distance = sys.float_info.max
        for i, cluster in enumerate(self.micro_clusters):
            center = cluster.get_center()
            for next_cluster in self.micro_clusters[i+1:]:
                next_cluster_center = next_cluster.get_center()
                # print(center)
                # print(next_cluster_center)
                dist = distance.euclidean(center, next_cluster_center)
                if dist < min_distance:
                    min_distance = dist
                    cluster_1 = cluster
                    cluster_2 = next_cluster
        assert (cluster_1 != cluster_2)
        cluster_1.merge(cluster_2)
        self.micro_clusters.remove(cluster_2)

    def break_cluster(self,X,y,cluster,IL,CurrentTime):
        Ra = 1 - IL
        if len(self.micro_clusters) >= maxMcs:
            self.merge_cluster()
        self.creat_new(X=X,y=y,cluster=cluster,Ra=Ra,CurrentTime=CurrentTime)
        if cluster.nb_points - self.micro_clusters[-1].nb_points==0:
            cluster.nb_points = 1
        else:
            cluster.nb_points = cluster.nb_points - self.micro_clusters[-1].nb_points
        cluster.Mc_center = cluster.Mc_center
        cluster.linear_sum = cluster.linear_sum - self.micro_clusters[-1].linear_sum
        cluster.squared_sum = cluster.squared_sum - self.micro_clusters[-1].squared_sum
        cluster.Mc_radius = cluster.Mc_radius - self.micro_clusters[-1].Mc_radius
        cluster.update_timestamp = CurrentTime
    def update_model(self,CurrentTime):
        for i in range(len(self.micro_clusters)-1):
            try:
                T = CurrentTime - self.micro_clusters[i].update_timestamp
                imps = self.micro_clusters[i].W
                imps = imps*2**(lamda*T)
                spe = self.micro_clusters[i].Mc_special
                spe = spe*2**(lamda1*T)
                self.micro_clusters[i].W = imps
                self.micro_clusters[i].Mc_special = spe
                cluster_r = self.micro_clusters[i]
                if imps < wT:
                    self.micro_clusters.remove(cluster_r)
            except IndexError:
                pass