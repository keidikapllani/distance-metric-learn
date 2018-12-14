#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 22:03:28 2018

@author: keidi
"""

import numpy as np
from src.myutils import load_data, dist_mat,cnc, knn_reid, estimate_reconstruction_error, load_images_from_folder, frame_image
from src.class_PersonReId import PersonReId
from re_ranking.re_rank import re_ranking


# Load CUHK03 dataset.
# Folder PR_data needs to be in the correct path.
folder = 'PR_data/cuhk03_new_protocol_config_labeled.mat'
camId,filelist,gallery_idx,labels,query_idx,train_idx,features = load_data(folder)

# Initialise PersonReId custom class
strain = PersonReId(train_idx,camId,labels,features,use_index = True)
squery = PersonReId(query_idx,camId,labels,features,use_index = True)
sgallery = PersonReId(gallery_idx,camId,labels,features,use_index = True)


'''
###############################################################################
#################### Feature Re-Ranking ##########################
'''
ng, dg = sgallery.features.shape
nq,dq = squery.features.shape
neighbours_list = []
indices_list = []
        
for l in range(1,7,1):                
    final_dist = re_ranking(squery.features,sgallery.features,20,6,l/10, MemorySave = False, Minibatch = 2000)
    
    indeces = np.zeros((nq,ng))	
    for i in range(0,nq):
    	# print completion
    	print(f'Completion {100*i/nq}%')
    	# Measure distances of query vector with each gallery vector
    	for j in range(0,ng):
    		# Set distance to infinity if query has same cameraID and label
    		if squery.camId[i] == sgallery.camId[j] and squery.labels[i] == sgallery.labels[j]:
    			final_dist[i,j] = np.inf			 
    
    sort_idx = np.argsort(final_dist, axis = 1)	
    # Sort picture index based on the sorted gallery features
    indices_list.append(sgallery.idx[sort_idx])
    neighbours_list.append(sgallery.labels[sort_idx])


#Measure the rank(k) of each iteration    
accuracy = np.zeros((500,6))
for j in range(6):
    for i in range(0,500):
    	acc, yknn = knn_reid(squery, neighbours_list[j], i+1 )
    	accuracy[i,j] = acc

print(f'Re_rank score at rank(1): {accuracy[0,0]} s')