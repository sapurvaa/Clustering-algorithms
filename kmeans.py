import pandas as pd
import numpy as np
import random
from scipy.spatial import distance
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.metrics import davies_bouldin_score

def initialize_centroid(dataset,k):
    random.seed(1000)
    centroids = random.sample(list(dataset), k)
    #print("initial centroids " + str(centroids))
    return centroids

def nearest_centroid(point,centroids):
    dist = []
    for centroid in centroids:
        dist.append(distance.euclidean(point,centroid))
    min_dist = np.array(dist)
    cluster = min_dist.argmin()
    return cluster

def mean_dist(cluster_points):
    new_centroid = np.mean(cluster_points, dtype=np.float64,axis=0)
    return new_centroid

def plot_clusters(dataset,k,cluster_index,centroids):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ["red", "green", "yellow", "blue", "pink", "purple", "orange", "black", "grey", "brown"]
    for point in range(len(cluster_index)):
        ax.scatter(dataset[point, 0], dataset[point, 1], dataset[point, 2], c=colors[cluster_index[point]])
    for centroid in range(len(centroids)):
        ax.scatter(centroids[centroid][0],centroids[centroid][1],centroids[centroid][2],c="maroon",marker="X")
    ax.set_xlabel("all_NBME_avg_n4")
    ax.set_ylabel("all_PIs_avg_n131")
    ax.set_zlabel("HD_final")
    ax.set_title("number of clusters = " + str(k))
    plt.show()

def davies_bouldin(dataset,k,cluster_index):
    #print("calculating DB index")
    ri = []
    for i in range(0,k):
        rij=[]
        for j in range(0,k):
            if (i != j):
                cluster_points_i = []
                cluster_points_j = []
                for x in range(len(cluster_index)):
                    if cluster_index[x] == i:
                        cluster_points_i.append(dataset[x,:])
                    elif cluster_index[x] == j:
                        cluster_points_j.append(dataset[x,:])
                si_sum = 0
                sj_sum = 0
                for m in cluster_points_i:
                    si_sum = si_sum + distance.euclidean(m,centroids[i])
                si = si_sum/len(cluster_points_i)
                for m in cluster_points_j:
                    sj_sum = sj_sum + distance.euclidean(m, centroids[j])
                sj = sj_sum/len(cluster_points_j)
                dij = distance.euclidean(centroids[i],centroids[j])
                rij.append((si + sj)/dij)
        ri.append(max(rij))
    db = sum(ri)/k
    return db

def plot_graph(db_indexes):
    cluster_num = []
    dbi_value = []
    for k,v in db_indexes.items():
        cluster_num.append(k)
        dbi_value.append(v)
    plt.plot(cluster_num,dbi_value)
    plt.ylabel('db index')
    plt.xlabel("number of clusters")
    plt.show()

if __name__ == '__main__':

    complete_data = pd.read_csv("BSOM_DataSet_revised.csv",usecols=("all_NBME_avg_n4","all_PIs_avg_n131","HD_final"))
    #print(complete_data)
    data = complete_data.to_numpy()
    #print(data)
    db_indexes = {}
    db_sklearn = {}
    #k=3
    for k in range(2,11):
        centroids = initialize_centroid(data, k)
        num_iterations = 1
        #print("value of k " + str(k))
        while(num_iterations < 10):
            #print("iteration number is " + str(num_iterations))
            cluster_index = []
            for point in data:
                cluster_index.append(nearest_centroid(point,centroids))
            new_centroids = []
            for c in range(0,k):
                cluster_points = []
                for x in range(len(cluster_index)):
                    if cluster_index[x] == c:
                        cluster_points.append(data[x,:])
                new_centroid = mean_dist(cluster_points)
                new_centroids.append(new_centroid)
            num_iterations = num_iterations + 1
            if np.array_equal(centroids,new_centroids):
                break
            centroids = new_centroids
            #print("new_centroid" + str(centroids))
        plot_clusters(data,k,cluster_index,centroids)
        dbi = davies_bouldin(data,k,cluster_index)
        db_indexes.update({k:dbi})
        #print("db_indexes are " + str(db_indexes))
        #db_sklearn[k] = davies_bouldin_score(data, cluster_index)
        #print(db_sklearn)
    plot_graph(db_indexes)
