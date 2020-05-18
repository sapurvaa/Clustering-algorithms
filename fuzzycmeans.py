import pandas as pd
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.metrics import davies_bouldin_score

def initialize_membership(dataset,k):
    membership_matrix = np.empty([np.size(dataset,0),k])
    x = 0
    while x < len(dataset):
        a = np.random.random(k)
        n = []
        for v in a:
            n.append(v/a.sum())
        membership_matrix[x,:] = n
        x=x+1
    #print("membership_matrix" + str(membership_matrix))
    return membership_matrix

def cluster_center(dataset,k,m,member_matrix):
    features = np.size(dataset,1)
    centroids = []
    for c in range(k):
        centre = []
        for feature in range(features):
            num = 0
            den = 0
            for x in range(len(dataset)):
                num = num + (np.power(member_matrix[x,c],m) * dataset[x, feature])
                den = den + (np.power(member_matrix[x,c],m))
            centre.append(num/den)
        centroids.append(np.array(centre))
    #print("centroids" + str(centroids))
    return centroids

def centroid_distance(dataset,centroids):
    distance = np.empty([np.size(dataset, 0), k])
    for x in range(len(dataset)):
        dist = []
        for c in range(k):
            point = dataset[x,:]
            centre = centroids[c]
            dist.append(np.linalg.norm(point-centre))
        distance[x,:] = dist
    #print("distance matrix is " + str(distance))
    return distance

def update_membership(distance_matrix,k,m):
    new_member_matrix = np.empty(distance_matrix.shape)
    for x in range(len(new_member_matrix)):
        for c in range(k):
            int_sum = 0
            for j in range(k):
                p = np.power((distance_matrix[x,c]/distance_matrix[x,j]),2)
                int_sum = int_sum + p
            n = np.reciprocal(int_sum)
            new_member_matrix[x,c] = np.array([n])
    #print("new member matrix is " + str(new_member_matrix))
    return new_member_matrix

def harden_cluster(member_matrix):
    cluster_index = []
    for x in range(len(member_matrix)):
        cluster_index.append(np.argmax(member_matrix[x,:]))
    return cluster_index

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
    print("calculating DB index")
    ri = []
    for i in range(0,k):
        rij = []
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

if __name__ == '__main__':

    complete_data = pd.read_csv("BSOM_DataSet_revised.csv", usecols=("all_NBME_avg_n4", "all_PIs_avg_n131", "HD_final"))
    #print(complete_data)
    data = complete_data.to_numpy()
    #print("dataset is " + str(data))
    k=3
    m=2
    num_iterations = 1
    member_matrix = initialize_membership(data,k)
    while num_iterations < 100:
        centroids = cluster_center(data,k,m,member_matrix)
        distance_matrix = centroid_distance(data,centroids)
        new_member_matrix = update_membership(distance_matrix,k,m)
        if(np.allclose(member_matrix,new_member_matrix)):
            break
        member_matrix = new_member_matrix
        num_iterations = num_iterations + 1
        #print("num_iterations" + str(num_iterations))

    cluster_index = harden_cluster(member_matrix)
    #plot_clusters(data,k,cluster_index,centroids)
    db = davies_bouldin(data,k,cluster_index)
    print("db index is " + str(db))
    #db_sklearn=davies_bouldin_score(data, cluster_index)
    #print(db_sklearn)
