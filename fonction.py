import random
import numpy as np
from collections import Counter
import math
import numpy as np
def transpose(matrix):
    if matrix == None or len(matrix) == 0:
        return []
        
    result = [[None for i in range(len(matrix))] for j in range(len(matrix[0]))]
    
    for i in range(len(matrix[0])):
        for j in range(len(matrix)):
            result[i][j] = matrix[j][i]
            
    return result

def compute_center(matrix):
    center = []
    for i in range (len(matrix)):
        center.append([])
        for column in transpose(matrix[i]):
            if str(type(column[0]))=="<class 'str'>":
                unique_values, counts = np.unique(np.array(column), return_counts=True)
                most_frequent_index = np.argmax(counts)
                center[i].append(unique_values[most_frequent_index])
            else:
                center[i].append(np.mean(np.array(column)))
    return np.array(center)


def euclidean_distance(x,y):
    s=0
    for i in range(len(x)):
        if str(type(x[i]))=="<class 'str'>":
            if str(x[i])!=str(y[i]):
                s=s+1
        else:
            s=s+((float(x[i])-float(y[i]))**2)
    
    return math.sqrt(s)

def is_nominal(data, col_name):
    col_num=[]
    col_nominal=[]
    for i in range(len(col_name)):
        col_type = data.dtypes[col_name[i]]
    
        if col_type == np.object: 
            col_nominal.append(col_name[i])
        else:  
            col_num.append(col_name[i])

    return col_num,col_nominal
    
def normaliser_data(data):
    data=np.array(data).T
    data_num=np.zeros(data.shape)
    for i in range(len(data)):
        min_val = min(data[i])
        max_val = max(data[i])
        for j in range(len(data[0])):
            data_num[i][j] = ((data[i][j]- min_val)/(max_val - min_val))

    return data_num.T

def seuil_distance_function(data):
    distance=[]
    n=len(data)
    for i in range(n):
        for j in range(i+1,n):
            d=euclidean_distance(data[i], data[j])
            distance.append(round(d,1))
    dic=dict(Counter(distance))
    distance_dic=sorted(dic.items(),key=lambda x:x[1],reverse=True) 
    distance=[]
    for i in range(len(distance_dic)):
        distance.append(distance_dic[i][0])

    distance_freq=distance[0]
    median=distance[int(len(distance)/2)]
    Prem_car=distance[int(len(distance)/4)]
    for i in range(1,len(distance)):
        if distance_freq <Prem_car or distance_freq>median:
            distance_freq=distance[i]
    return distance_freq

def k_auto(data,seuil_distance):
    used_index=[]
    cluster_indexes=[]
    n=len(data)
    x=n
    i=0
    k=0
    while i<n:
        individu_aliatoir=random.randint(0, x-1)
        
        while individu_aliatoir in used_index:
            individu_aliatoir=random.randint(0, x-1)
        
        used_index.append(individu_aliatoir)
        cluster_indexes.append([individu_aliatoir])
        n=n-1
        j=0
        n1=n
        while j<n1:
            index_data=random.randint(0, x-1) 
            while index_data in used_index:
                index_data=random.randint(0, x-1)
            x1=data[index_data]
            y=data[individu_aliatoir]
            distance=euclidean_distance(x1,y)
            if distance <= seuil_distance:
                cluster_indexes[k].append(index_data)
                used_index.append(index_data)
                n=n-1
                
            n1=n1-1
        k=k+1
    
    return len(cluster_indexes)


def kmeans(k,data,nb_iteration):
    n=len(data)
    m=len(data[0])

    #initialisation des cluster selon le k
    cluster_initial=[]
    for i in range(k):
        cluster_initial.append([])
    
    liste=[]
    x=0
    nb_cluster=k
    for _ in range(n):
        i=random.randint(0, n-1)
        while i in liste:
            i=random.randint(0, n-1)
        liste.append(i)
        cluster_initial[x].append(data[i])
        x=x+1
        if x==nb_cluster:
            x=0
        
    #les centre des cluster
    centre_cluster=compute_center(cluster_initial)
    print(centre_cluster)
    
    #traitement 
    centre_cluster_final=[]
    while  (not np.array_equal(np.array(centre_cluster),np.array(centre_cluster_final))) or (nb_iteration >0):
        for i in range(n):
            distance=[]
            for j in range(k):
                d=euclidean_distance(data[i],centre_cluster[j])
                distance.append((d,j))
        
            distance=sorted(distance,key=lambda x:x[0])
            indice_cluster=distance[0][1]
    

            if data[i] not in cluster_initial[indice_cluster]:
                for l in range (len(cluster_initial)):
                    if data[i] in cluster_initial[l] :
                        cluster_initial[l].remove(data[i])  
                
                cluster_initial[indice_cluster].append(data[i])

        delete=[]
        for h in range(len(cluster_initial)):
            if(len(cluster_initial[h])==0):
                delete.append(h)
        for h in delete:
            cluster_initial.pop(h)
            nb_cluster=nb_cluster-1
            k=k-1
        centre_cluster_final=centre_cluster
        centre_cluster=compute_center(cluster_initial)
        print(centre_cluster)
        nb_iteration=nb_iteration-1
        
    return cluster_initial

