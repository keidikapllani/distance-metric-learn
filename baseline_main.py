# -*- coding: utf-8 -*-
"""
Script to conduct baseline measurments.
EE4_68 Pattern Recognition - Imperial College London EEE Dept.

Created on Tue Nov 27 2018
@authors: Antonio Enas, Keidi Kapllani
"""
# Import libraries
import numpy as np
from src.myutils import load_data, dist_mat,cnc, knn_reid, estimate_reconstruction_error, load_images_from_folder, frame_image
from sklearn.decomposition import PCA  
from src.class_PersonReId import PersonReId
from sklearn.cluster import KMeans
import scipy.spatial.distance as spd
import time
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
#%matplotlib auto

# Load CUHK03 dataset.
# Folder PR_data needs to be in the correct path.
folder = 'PR_data/cuhk03_new_protocol_config_labeled.mat'
camId,filelist,gallery_idx,labels,query_idx,train_idx,features = load_data(folder)

# Initialise PersonReId custom class
strain = PersonReId(train_idx,camId,labels,features,use_index = True)
squery = PersonReId(query_idx,camId,labels,features,use_index = True)
sgallery = PersonReId(gallery_idx,camId,labels,features,use_index = True)

'''
################## Baseline kNN with Euclidean Distance #######################
Evaluation of baseline method.
'''
# Start time for time performance evaluation
_base_t_start = time.time()
# Rank knn labels and pic indexes, retrieve distance matrix
base_labels,base_dist_mat,base_idx = dist_mat(squery,sgallery,'euclidean')
base_cnc = cnc(squery,base_labels,20)
# Baseline total runtime
base_runtime = (time.time() - _base_t_start)
print(f'Baseline knn runtime before dimension reduction: {base_runtime} s')
print(f'Baseline knn score at rank(1): {base_cnc} s')

'''
################################## PCA ########################################
Improvement: dimensionality reduction to avoid "curse of dimensionality" and
speed up computation.
We find a suitable M for compression based on theoretical reconstruction error.
PCA fit on TRAINING data and transform applied to query and gallery.
'''
# Estimate PCA theoretical reconstruction error in training set
J_theor = estimate_reconstruction_error(strain)

# Apply PCA for dimensionality reduction with M = 100 components, J = 5%
pca = PCA(n_components = 100)
strain_reduced = strain
strain_reduced.features = pca.fit_transform(strain_reduced.features)
sgallery_reduced = sgallery
sgallery_reduced.features = pca.transform(sgallery_reduced.features)
squery_reduced = squery
squery_reduced.features = pca.transform(squery_reduced.features)

''' Perform baseline kNN performance after PCA dimension reduction '''
# Start time for time performance evaluation
_base_pca_t_start = time.time()
# Rank knn labels and pic indexes, retrieve distance matrix
base_pca_labels,base_pca_dist_mat,base_pca_idx = dist_mat(squery,
														  sgallery,'euclidean')
base_pca_cnc = cnc(squery_reduced,base_pca_dist_mat,20)
# Baseline total runtime
base_runtime = (time.time() - _base_pca_t_start)
print(f'Baseline knn runtime after dimension reduction: {base_runtime} s')
print(f'Baseline knn score at rank(1): {base_cnc[0]} ')


'''
################################# KMEANS ######################################
Perform k-means clustering 
'''
k = 8
kmeans = KMeans(n_clusters=700)
kmeans.fit(sgallery.features)
prediction = kmeans.predict(squery.features)
centroids = kmeans.cluster_centers_
clust_group = kmeans.labels_
y_knn = np.zeros((squery.n,k))
for i in range(0,squery.n):
    indx = np.column_stack(np.where(prediction[i]==clust_group))
    distance = np.zeros((len(indx[:,0])))
    clusters = sgallery.features[indx[:,0]]
    clus_labs = sgallery.labels[indx[:,0]]
    clust_cams = sgallery.camId[indx[:,0]]
    for j in range(0,len(indx)):
        if squery.labels[i] == clus_labs[j] and  squery.camId[i] == clust_cams[j]:
            distance[j] = np.inf
        else:
            distance[j] = spd.pdist(np.array([centroids[prediction[i],:],clusters[j,:]]),metric = 'euclidean')
    sort_idx = np.argsort(distance)
    sorted_clusters = clusters[sort_idx]
    	    
    for k in range(1,8):
        if squery.labels[i] in sgallery.labels[indx[sort_idx[:k]]]:
            
            y_knn[i,k-1] = True
        else:
            y_knn[i,k-1] = False


#accuracy k means
accuracy = np.zeros((k))
for i in range(k):
    accuracy[i] = np.sum(y_knn[:,i])/1400

#PLotting CNC for k-means
  
print(f'Baseline k-means score at rank(1): {accuracy[0]} ')

plt.figure()
plt.title('CNC of k-means clustering',fontsize=18)
plt.xlabel('Number of neighbours - k',fontsize=14)
plt.ylabel('Rank(k)',fontsize=14)
plt.plot(accuracy)
plt.grid(True)    
'''
############### Baseline approach: KNN with different metrics #################
Vary k neirest neighbours and metric used.
'''

accuracy = np.zeros((5001,14))
neighbours_labels = np.zeros((14,squery.n,sgallery.n))
distances = np.zeros((14,squery.n,sgallery.n))
indeces = np.zeros((14,squery.n,sgallery.n))
time_metric = np.zeros((14,))
METRICS = ['euclidean','cityblock', 'correlation','cosine','jaccard',
		   'braycurtis', 'canberra', 'chebyshev',
		   'hamming', 'minkowski','sqeuclidean','seuclidean']
mct = 0
for m in METRICS:
	ts = time.time()
	neighbours_labels[mct,:,:],distances[mct,:,:],indeces[mct,:,:] = dist_mat(squery, sgallery, m)
	time_metric[mct] = time.time() - ts
	for k in range(1,5000):
		acc, yknn = knn_reid(squery, neighbours_labels[mct,:,:], k )
		accuracy[k-1,mct] = acc
	mct += 1


'''
#################### Plotting Results ##########################
'''

'''
%%%%%%%%
Plotting CNC for knn baseline
%%%%%%%%
'''
METRICS = ['euclidean','jaccard',
		   'minkowski']
xint = range(1,21)
plt.figure()
plt.title('kNN baseline for different metrics',fontsize=18)
plt.xlabel('Number of neighbours - k',fontsize=14)
plt.ylabel('Rank(k)',fontsize=14)
for i in range(len(METRICS)):
    plt.plot(xint,accuracy[:20,i],label=METRICS[i],linewidth=2.0)
    
plt.legend(ncol=1,loc=7,fontsize=14)
plt.xticks(xint)
plt.axis('tight')
plt.yticks(np.arange(0, 100, step=10))
plt.axis('tight')
plt.grid(True)

'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Plot ranklist for knn baseline
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

images = load_images_from_folder('./PR_data/images_cuhk03/')
indeces= indeces.astype(int)
plt.figure(figsize = (5,10))
gs1 = gridspec.GridSpec(5, 10)
gs1.update(wspace=0, hspace=0.015, left= 0.01, right =1) # set the spacing between axes. 

for i in range(5):
    ax1 = plt.subplot(gs1[i*10])
    ax1.imshow(frame_image(images[squery.idx[i]],5,'black'),interpolation='none')
    plt.axis('off')
    for j in range(1,10):
       # i = i + 1 # grid spec indexes from 0
        ax1 = plt.subplot(gs1[i*10+j])
        if squery.labels[i] == neighbours_labels[i,j-1]:
            ax1.imshow(frame_image(images[indeces[i,j-1]],5,'green'),interpolation='none')
        else:
            ax1.imshow(frame_image(images[indeces[i,j-1]],5,'red'),interpolation='none')
    
        plt.axis('off')
    
plt.show()

