# -*- coding: utf-8 -*-
"""
Utility functions for coursework 2, pattern recognition.

Created on Tue Nov 27 10:25:34 2018
@authors: Antonio Enas, Keidi Kapllani
"""

import numpy as np
import scipy.io as sio
import scipy.spatial.distance as spd
from sklearn.cluster import KMeans
import sklearn.utils.linear_assignment_ as la
from sklearn.metrics import accuracy_score
import json
import os
import torch as pt
from os import listdir
from PIL import Image as PImage
from collections import defaultdict



def load_data(folder):
	'''
	load_data imports the data needed for the coursework
	'''
	# Unpack .MAT file
	mat_content = sio.loadmat(folder)

	# Reassign variable names
	
	camId = mat_content['camId'].flatten()	
	#camId specifies whether image was taken from camera 1 or camera 2. During
	#testing do not consider images of your current query from the same camera.
	
	filelist = mat_content['filelist'].flatten()
	#filelist specifies correspondences between names of files in
	#images_cuhk03 and their indexes.
	#access as filelist[idx][0][0]
	
	gallery_idx = np.array(mat_content['gallery_idx'].flatten()-1,int)
	#gallery_idx specifies indexes of the part of the dataset from which
	#you compose your ranklists during testing phase
	
	labels = mat_content['labels'].flatten()
	#labels contains ground truths for each image
	
	query_idx = np.array(mat_content['query_idx'].flatten()-1,int)
	#query_idx contains indexes of query images
	
	train_idx = np.array(mat_content['train_idx'].flatten()-1,int)
	#train_idx contains indexes of images to be used for training and validation
	
	# Import the features
	json_filepath = os.path.join('.', 'PR_data', 'feature_data.json')
	with open(json_filepath, 'r') as f:
		features = json.load(f)
	features = np.array(features) 
	return camId,filelist,gallery_idx,labels,query_idx,train_idx,features

def dist_mat(query, gallery, distance_metric = 'euclidean'):
	'''
	KNN fit to purpose implementation for CHUK03 protocol testing.
	Set distance of gallery vector with same cameraID and label to Inf.
	'''
	# Dimensions of query and gallery
	ng, dg = gallery.features.shape
	nq,dq = query.features.shape
	# Initialise distances and picture index matrices
	distances = np.zeros((nq,ng))
	indeces = np.zeros((nq,ng))
	
	# KNN algorithm with exclusion
	# For each query
	for i in range(0,nq):
		# print completion
		print(f'Completion {100*i/nq}%')
		# Measure distances of query vector with each gallery vector
		for j in range(0,ng):
			# Set distance to infinity if query has same cameraID and label
			if query.camId[i] == gallery.camId[j] and query.labels[i] == gallery.labels[j]:
				distances[i,j] = np.inf			
			else:
				distances[i,j] = spd.pdist(np.array([query.features[i,:],gallery.features[j,:]]),metric = distance_metric)
	
	# Retrieve index of sorted gallery vectors		
	sort_idx = np.argsort(distances, axis = 1)	
	# Sort picture index based on the sorted gallery features
	for i in range(0,nq):
		indeces[i,:] = gallery.idx[sort_idx[i,:]]
	
	# Collect k neighbours' labels
	neighbours_labels = np.zeros((nq,ng))
	for i in range(0,ng-10):
		neighbours_labels[:,i] = gallery.labels[sort_idx[:,i]]
	return neighbours_labels,distances,indeces

def knn_reid(query, neighbours_labels, k = 1):
	'''
	Attempts identity retrieval within the k nearest neighbours.
	Computes CNC at rank k.
	Inputs:
		query : PersonReId type
		neighbours_labels : matrix containing sorted labels ~ np.array
		k : nearest neighbours, limit for retrieval estimation ~ int
	'''
	yknn = np.zeros((query.n,))
	for i in range(0,query.n):
		if query.labels[i] in neighbours_labels[i,0:k]:
			yknn[i] = query.labels[i]
		else:
			yknn[i] = neighbours_labels[i,0]
	
	# Estimate prediction accuracy
	accuracy = 100*accuracy_score(yknn, query.labels)
	return accuracy, yknn

def cnc(query,neighbours_labels,k):
	'''
	Evaluates the CNC up to k nearest neighbours.
	'''
	cnc = np.zeros((k,))
	for i in range(0,k):
		cnc[i], yknn = knn_reid(query, neighbours_labels, i+1 )
	return cnc

def estimate_reconstruction_error(train):
	'''
	Estimate theoretical reconstruction error of PCA training data.
	Input:
		train : PersonReId type
	'''
	# Initialise variables
	N = train.n	#number of samples
	meanfeature = train.features.mean(axis = 0)
	A = train.features - meanfeature
	# Covariance matrix
	S = (1 / N) * np.dot(A.T, A) # D*D matrix
	wn, U = np.linalg.eig(S)
	U = np.real(U)
	# Here we find D eigenval and eigenvectors
	w_n = sorted(np.real(wn), reverse=True)
	# Initialise variables
	J_theor = np.zeros((2048,),float)
	eigsum 	= sum(w_n) #Total sum of the eigenvalues
	# Vary M from 0 to D and estimate the theoretical reconstruction error
	for m in range(0,2048):
		J_theor[m] = 100*(eigsum - sum(w_n[:m]))/eigsum
	return J_theor



def pair_idx(train_data):
	'''
	Generate two index lists, a list of positive (same identity but different
	camera) and a list of negative pairs (different identity) to facilitate
	metric learning with KISSME.
	'''
	#initialise output variables
	negative_id = []
	positive_id = []
	# for each data point
	for n in range(0,len(train_data.idx)):
		# Print completion status
		c = 100*n/train_data.n
		print(f'{c}% Completed')
		
		idx1 = train_data.idx[n]
		# combine with all other pictures but itself
		for i in range(n+1,len(train_data.idx)):	
			
			# if not same identity match the pair in the negatives
			if train_data.labels[n] != train_data.labels[i]:
				negative_id.append((idx1,train_data.idx[i]))
			
			# else if same label and different cam append in the positives
			elif train_data.labels[n] == train_data.labels[i] and train_data.camId[n] != train_data.camId[i]:				
				positive_id.append((idx1,train_data.idx[i]))
			
			# else ignore
	return positive_id, negative_id

def sum_cov(train,id_pair,optimise=0):
	'''
	Compute summation{(x1-x2)(x1-x2).T}, over each pair (x1,x2).
	'''
	# initialise variance matrix
	d = len(train.features[0])
	s = np.zeros((d,d))
	if optimise == 0:
		# for each index pair
		for i in range(0,len(id_pair)):
			# Print completion status
			c = 100*i/len(id_pair)
			print(f'{c}% Completed')
			
			# compute the matrix
			s += np.dot((train.features_of(id_pair[i][0])-train.features_of(id_pair[i][1])).T,(train.features_of(id_pair[i][0])-train.features_of(id_pair[i][1])))
		s = s/len(id_pair)
	else:
		# for each index pair
		for i in range(0,len(id_pair)):
			# Print completion status
			c = 100*i/len(id_pair)
			print(f'{c}% Completed')
			
			# compute the matrix
			s += pt.mm(pt.tensor((train.features_of(id_pair[i][0])-train.features_of(id_pair[i][1])).T),pt.tensor((train.features_of(id_pair[i][0])-train.features_of(id_pair[i][1]))))
		s = np.array(s)/len(id_pair)
	return s


def binary_class(train):
	'''
	Pair true positives and true negatives together to create training data for
	the neuronal network and kernel transformer.
	'''
	#initialise output variables
	negatives = []
	positives = []
	# for each data point
	for n in range(0,train.n):
		# Completion status
		c = 100*n/train.n
		print(f'{c}% Completed')
		
		# combine with all other pictures but itself
		app = train.features[n]
		for i in range(n,train.n):	
			
			# if not same identity match the pair in the negatives
			if train.labels[n] != train.labels[i]:
				 
				negatives.extend([[app],[train.features[i]]] )
			
			# else if same label and different camer append in the positives
			elif train.labels[n] == train.labels[i] & train.camId[n] != train.camId[i]:
				
				positives.extend([[app],[train.features[i]]])
			
			# else ignore
				
	
	return positives, negatives

def pair_positives(train):
	'''
	Pair true positives feature vectors to create training data for
	the neuronal network.
	'''
	#initialise output variables
	positives = []
	indexes = []
	# for each data point
	for n in range(0,train.n):
		# Completion status
		c = 100*n/train.n
		print(f'{c}% Completed')
		
		# combine with all other pictures but itself
		app = train.features[n]
		app_id = train.idx[n]
		for i in range(0,train.n):	
			
			# if not same identity match the pair in the negatives
			if train.labels[n] == train.labels[i] & train.camId[n] != train.camId[i]:
				positives.extend([[app],[train.features[i]]])
				indexes.extend(zip(app_id,train.idx[i]))
			# else ignore
					
	return positives, indexes

def eval_cuhk03(distmat, query, gallery, max_rank):
	"""Evaluation with cuhk03 metric
	Key: one image for each gallery identity is randomly sampled for each query identity.
	Random sampling is performed num_repeats times.
	"""
	q_pids = query.labels
	g_pids = gallery.labels
	q_camids = query.camId
	g_camids = gallery.camId
	num_repeats = 10
	num_q, num_g = distmat.shape
    
	indices = np.argsort(distmat, axis=1)
	matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
	all_cmc = []
	all_AP = []
	num_valid_q = 0. # number of valid query
    
	for q_idx in range(num_q):
        # get query pid and camid
		q_pid = q_pids[q_idx]
		q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
		order = indices[q_idx]
		remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
		keep = np.invert(remove)

        # compute cmc curve
		raw_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
		if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
			continue

		kept_g_pids = g_pids[order][keep]
		g_pids_dict = defaultdict(list)
		for idx, pid in enumerate(kept_g_pids):
			g_pids_dict[pid].append(idx)

		cmc, AP = 0., 0.
		for repeat_idx in range(num_repeats):
			mask = np.zeros(len(raw_cmc), dtype=np.bool)
			for _, idxs in g_pids_dict.items():
                # randomly sample one image for each gallery person
				rnd_idx = np.random.choice(idxs)
				mask[rnd_idx] = True
			masked_raw_cmc = raw_cmc[mask]
			_cmc = masked_raw_cmc.cumsum()
			_cmc[_cmc > 1] = 1
			cmc += _cmc[:max_rank].astype(np.float32)
            # compute AP
			num_rel = masked_raw_cmc.sum()
			tmp_cmc = masked_raw_cmc.cumsum()
			tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
			tmp_cmc = np.asarray(tmp_cmc) * masked_raw_cmc
			AP += tmp_cmc.sum() / num_rel
        
		cmc /= num_repeats
		AP /= num_repeats
		all_cmc.append(cmc)
		all_AP.append(AP)
		num_valid_q += 1.

	assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

	all_cmc = np.asarray(all_cmc).astype(np.float32)
	all_cmc = all_cmc.sum(0) / num_valid_q
	mAP = np.mean(all_AP)

	return all_cmc, mAP

def dist_mahalanobis(query, gallery,M):
	'''
	Evaluate the Mahalanobis distance given the covariance matrix M.
	'''
	# Dimensions of query and gallery
	ng, dg = gallery.features.shape
	nq,dq = query.features.shape
	# Initialise distances and picture index matrices
	distances = np.zeros((nq,ng))
	indeces = np.zeros((nq,ng))
	
	# KNN algorithm with exclusion
	# For each query
	for i in range(0,nq):
		# print completion
		print(f'Completion {100*i/nq}%')
		# Measure distances of query vector with each gallery vector
		for j in range(0,ng):
			# Set distance to infinity if query has same cameraID and label
			if query.camId[i] == gallery.camId[j] and query.labels[i] == gallery.labels[j]:
				distances[i,j] = np.inf			
			else:
				pair_x  = query.features[i,:] - gallery.features[j,:]
				distances[i,j] = np.linalg.multi_dot([pair_x,M,pair_x.T])
	
	# Retrieve index of sorted gallery vectors		
	sort_idx = np.argsort(distances, axis = 1)	
	# Sort picture index based on the sorted gallery features
	for i in range(0,nq):
		indeces[i,:] = gallery.idx[sort_idx[i,:]]
	
	# Collect k neighbours' labels
	neighbours_labels = np.zeros((nq,ng))
	for i in range(0,ng-10):
		neighbours_labels[:,i] = gallery.labels[sort_idx[:,i]]
	return neighbours_labels,distances,indeces


def mma(X,pairs,label,metrics):
	'''
	Mixed Metric Approach feature generator.
		X, feature data matrix (n_samples,dimension).
		pair, indexes to allow pairwise fusion in the new feature vector.
		label, new feature matrix labels
		metrics, array like ['euclidean','cosine','Jacard']
	'''
	df = len(metrics) 	#new feature matrix dimension
	nf = len(pairs)		#new feature matrix sample number
	
	# Initialise new feature matrix output
	F = np.zeros((nf,df))
	
	# Generate new feature vectors for each pair
	for p in range(0,nf):
		xi = X.features_of(pairs[p][0])
		xj = X.features_of(pairs[p][1])
		# Evaluate each of the chosen metrics
		for m in range(0,df):
			F[p,m] = spd.cdist(xi,xj,metric = metrics[m])
		rem = 100-100*p/nf
		print(f'Remaining: {rem}')
	
	# Generate the labels array
	Y = np.array([label for i in range(0,nf)])
	
	return F,Y


def mm_test(query, gallery):
	'''
	Evaluate the Mahalanobis distance given the covariance matrix M.
	'''
	# Dimensions of query and gallery
	ng, dg = gallery.features.shape
	nq,dq = query.features.shape
	# Initialise distances and picture index matrices
	Y = np.zeros((nq,ng))
	features = np.zeros((2,nq,ng))
	
	# KNN algorithm with exclusion
	# For each query
	for i in range(0,nq):
		# print completion
		print(f'Completion {100*i/nq}%')
		# Measure distances of query vector with each gallery vector
		for j in range(0,ng):
			
			if not(query.camId[i] == gallery.camId[j] and query.labels[i] == gallery.labels[j]):
				u = query.features[i,:]
				v = gallery.features[j,:]
				features[0,i,j] = np.linalg.norm(u - v) # euclidean
				features[1,i,j] = 1.0 - np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)) #cosine
				Y[i,j] = query.labels[i] == gallery.labels[j]

	return features,Y


def mma_rank(query, gallery, metrics):
	'''
	Create a new feature vector for each query gallery pair and use its magnitude
	to rank.
	The feature vector consist of different distances.
	'''
	# Dimensions of query and gallery
	ng, dg = gallery.features.shape
	nq,dq = query.features.shape
	# Initialise distances and picture index matrices
	distances = np.zeros((nq,ng,len(metrics)))
	indeces = np.zeros((nq,ng))
	#new_feature = np.zeros((len(metrics),))
	
	# KNN algorithm with exclusion
	# For each query
	for i in range(0,nq):
		# print completion
		print(f'Completion {100*i/nq}%')
		# Measure distances of query vector with each gallery vector
		for j in range(0,ng):
			# Set distance to infinity if query has same cameraID and label
			for m in range(0,len(metrics)):
				if query.camId[i] == gallery.camId[j] and query.labels[i] == gallery.labels[j]:
					distances[i,j,m] = np.inf			
				else:
					xi = query.features[i].reshape((1,dg))
					xj = gallery.features[j].reshape((1,dg))				
					distances[i,j,m] = spd.cdist(xi,xj, metric = metrics[m])
	# Rescale distances
	for m in range(0,len(metrics)):
		#new_feature[m] = spd.cdist(xi,xj, metric = metrics[m])
				
		distances[:,:,m] = (distances[:,:,m] - np.mean(distances[:,:,m],axis = 1).reshape((nq,1)))/(np.max(distances[:,:,m]) - np.min(distances[:,:,m]))
		
	# Retrieve index of sorted gallery vectors		
	sort_idx = np.argsort(distances, axis = 1)	
	# Sort picture index based on the sorted gallery features
	for i in range(0,nq):
		indeces[i,:] = gallery.idx[sort_idx[i,:]]
	
	# Collect k neighbours' labels
	neighbours_labels = np.zeros((nq,ng))
	for i in range(0,ng-10):
		neighbours_labels[:,i] = gallery.labels[sort_idx[:,i]]
	return neighbours_labels,distances,indeces


def load_images_from_folder(folder):
    imagesList = listdir(folder)
    loadedImages = []
    imagesList.sort()
    for image in imagesList:
        img = np.array(PImage.open(folder + image).resize((70,128)))
        loadedImages.append(img)

    return loadedImages

def frame_image(img, frame_width, colour):
    b = frame_width # border size in pixel
    ny, nx = img.shape[0], img.shape[1] # resolution / number of pixels in x and y
     # rgb or rgba array
    framed_img = np.zeros((b+ny+b, b+nx+b, 3)).astype(np.uint8)
    if colour == 'green':
        framed_img[:,:,1] = 255
    elif colour == 'red':
        framed_img[:,:,0] = 255
    elif colour == 'black':
        framed_img[:,:,0] = 0
        framed_img[:,:,1] = 0
        framed_img[:,:,2] = 0
    framed_img[b:-b, b:-b,0] = img[:,:,0]
    framed_img[b:-b, b:-b,1] = img[:,:,1]
    framed_img[b:-b, b:-b,2] = img[:,:,2]
    
    return framed_img

def dist_mat(query, gallery, distance_metric = 'euclidean'):
	'''
	KNN fit to purpose implementation for CHUK03 protocol testing.
	Set distance of gallery vector with same cameraID and label to Inf.
	Input:
		query : PersonReId type
		gallery : PersonReId type
		distance_metric : string
	'''
	# Dimensions of query and gallery
	ng, dg = gallery.features.shape
	nq,dq = query.features.shape
	# Initialise distances and picture index matrices
	distances = np.zeros((nq,ng))
	indeces = np.zeros((nq,ng))
	
	# KNN algorithm with exclusion
	# For each query
	for i in range(0,nq):
		# print completion
		print(f'Completion {100*i/nq}%')
		# Measure distances of query vector with each gallery vector
		for j in range(0,ng):
			# Set distance to infinity if query has same cameraID and label
			if query.camId[i] == gallery.camId[j] and query.labels[i] == gallery.labels[j]:
				distances[i,j] = np.inf			
			else:
				distances[i,j] = spd.pdist(np.array([query.features[i,:],gallery.features[j,:]]),metric = distance_metric)
	
	# Retrieve index of sorted gallery vectors		
	sort_idx = np.argsort(distances, axis = 1)	
	# Sort picture index based on the sorted gallery features
	for i in range(0,nq):
		indeces[i,:] = gallery.idx[sort_idx[i,:]]
	
	# Collect k neighbours' labels
	neighbours_labels = np.zeros((nq,ng))
	for i in range(0,ng-10):
		neighbours_labels[:,i] = gallery.labels[sort_idx[:,i]]
	return neighbours_labels,distances,indeces

def pair_constraint(train_data):
	'''
	Generate two index lists, a list of positive (same identity but different
	camera) and a list of negative pairs (different identity) to facilitate
	metric learning with KISSME.
	'''
	#initialise output variables
	negative_id = []
	positive_id = []
	# for each data point
	for n in range(0,len(train_data.idx)):
		# Print completion status
		c = 100*n/train_data.n
		print(f'{c}% Completed')
		
		idx1 = n
		# combine with all other pictures but itself
		for i in range(n+1,len(train_data.idx)):	
			
			# if not same identity match the pair in the negatives
			if train_data.labels[n] != train_data.labels[i]:
				negative_id.append((idx1,i))
			
			# else if same label and different cam append in the positives
			elif train_data.labels[n] == train_data.labels[i] and train_data.camId[n] != train_data.camId[i]:				
				positive_id.append((idx1,i))
			
			# else ignore
			
	pos = np.array(positive_id)
	neg = np.array(negative_id)
	
	np.random.seed(0)
	mask = np.random.choice(range(0,len(neg)), len(pos), replace=False)
	mask = np.sort(mask)
	tst_neg_id =[neg[i] for i in mask]
	neg = np.array(tst_neg_id)
	
	
	return pos[:,0],pos[:,1],neg[:,0],neg[:,1]

def pair_mmnn(Y):
	'''
	Generate two index lists, a list of positive (same identity but different
	camera) and a list of negative pairs (different identity) to facilitate
	metric learning with KISSME.
	'''
	#initialise output variables
	negative_id = []
	positive_id = []
	# for each data point
	for n in range(0,len(Y)):
		# Print completion status
		c = 100*n/len(Y)
		print(f'{c}% Completed')
		
		# combine with all other pictures but itself
		for i in range(n+1,len(Y)):	
			
			# if not same label
			if Y[n] != Y[i]:
				negative_id.append((n,i))
			
			# else if same label
			elif Y[n] == Y[i]:
				positive_id.append((n,i))
			
			# else ignore
			
	pos = np.array(positive_id)
	neg = np.array(negative_id)
	
	np.random.seed(0)
	mask = np.random.choice(range(0,len(neg)), len(pos), replace=False)
	mask = np.sort(mask)
	tst_neg_id =[neg[i] for i in mask]
	neg = np.array(tst_neg_id)
	
	
	return pos[:,0],pos[:,1],neg[:,0],neg[:,1]


def mma_proba_rank(query, gallery,metrics, mlp):
	'''
	Evaluate and sort query-gallery pairwise distance with Mixed Metric NN.
	For each pair generate a new feature vector containing their distance containing
	multiple distances.
	Combine the different distances into one with the matrix learned from the MLP.
	
	Inputs:
		query = query data contained in a PersonReId class
		gallery = gallery data contained in a PersonReId class
		metrics = list of metrics, e.g. ['euclidean','cosine','braycurtis','canberra']
		MNN1 = new feature projection matrix learned with MLP
		MNN2 = projected features weights learned with MLP
	Outputs:
		neighbours_labels = ranked predictions
		distances = ranked distances
		indeces = indexes of the pictures ranked
	'''
	# Dimensions of query and gallery
	ng, dg = gallery.features.shape
	nq,dq = query.features.shape
	# Initialise distances and picture index matrices
	distances = np.zeros((nq,ng))
	indeces = np.zeros((nq,ng))
	
	# Initialise new feature vector containing a different pairwise
	f = np.zeros((len(metrics),))
	
	# For each query
	for i in range(0,nq):
		# print completion
		print(f'Completion {100*i/nq}%')
		
		# Find query-gallery pair distance
		for j in range(0,ng):
			
			# Set distance to infinity if query has same cameraID and label
			if query.camId[i] == gallery.camId[j] and query.labels[i] == gallery.labels[j]:
				distances[i,j] = 1
			else:
				# Generate new feature vector for the pair						
				for m in range(0,len(metrics)):
					xi = query.features[i].reshape((1,dg))
					xj = gallery.features[j].reshape((1,dg))
					f[m]  = spd.cdist(xi,xj, metric = metrics[m])
				distances[i,j] = mlp.predict_proba(f.reshape((1,4)))[0,0]
		
	# Retrieve index of sorted gallery vectors		
	sort_idx = np.argsort(distances, axis = 1)	
	# Sort picture index based on the sorted gallery features
	for i in range(0,nq):
		indeces[i,:] = gallery.idx[sort_idx[i,:]]
	
	# Collect k neighbours' labels
	neighbours_labels = np.zeros((nq,ng))
	for i in range(0,ng-10):
		neighbours_labels[:,i] = gallery.labels[sort_idx[:,i]]
	
	return neighbours_labels,distances,indeces

def mma_rank(query, gallery,mlp):
	'''
	Evaluate and sort query-gallery pairwise distance with Mixed Metric NN.
	For each pair generate a new feature vector containing their distance containing
	multiple distances.
	Combine the different distances into one with the matrix learned from the MLP.
	
	'''
	# Dimensions of query and gallery
	ng, dg = gallery.features.shape
	nq,dq = query.features.shape
	# Initialise distances and picture index matrices
	distances = np.zeros((nq,ng))
	indeces = np.zeros((nq,ng))
	
	# Initialise new feature vector containing a different pairwise
	
	
	# For each query
	for i in range(0,nq):
		# print completion
		print(f'Completion {100*i/nq}%')
		
		# Find query-gallery pair distance
		for j in range(0,ng):
			
			# Set distance to infinity if query has same cameraID and label
			if query.camId[i] == gallery.camId[j] and query.labels[i] == gallery.labels[j]:
				distances[i,j] = np.inf
			else:
				xj = mlp.predict(gallery.features[j].reshape((1,150)))
				distances[i,j] = spd.cdist(query.features[i].reshape((1,150)),xj,'braycurtis')
		
	# Retrieve index of sorted gallery vectors		
	sort_idx = np.argsort(distances, axis = 1)	
	# Sort picture index based on the sorted gallery features
	for i in range(0,nq):
		indeces[i,:] = gallery.idx[sort_idx[i,:]]
	
	# Collect k neighbours' labels
	neighbours_labels = np.zeros((nq,ng))
	for i in range(0,ng-10):
		neighbours_labels[:,i] = gallery.labels[sort_idx[:,i]]
	
	return neighbours_labels,distances,indeces


from copy import copy
def ae_recall(query,sorted_labels,k):
	binary_mat = copy(sorted_labels[:,:-10])
	# for each query
	for q in range(0,query.n):
		# set to 1 the matching label and to 0 the otherse
		for k in range(0,binary_mat.shape[1]):
			if binary_mat[q,k] == query.labels[q]:
				binary_mat[q,k] = 1
			else:
				binary_mat[q,k] = 0
	# find number true positives
	true_p = np.sum(binary_mat,axis = 1)
	
	all_recall = np.zeros(binary_mat.shape)
	# determine recall
	for r in range(0,len(binary_mat.T)):
		all_recall[:,r] = np.sum(binary_mat[:,:r],axis = 1)/true_p
	tot_recall = all_recall.mean(axis = 0)
	
	# determine cnc and precision
	cnc = np.zeros(len(binary_mat.T,))
	avg_precision =  np.zeros(len(binary_mat.T,))
	avg_precision[0] = 100
	cnc[0] = 0
	for i in range(1,len(binary_mat.T)):
		cnc[i], yknn = knn_reid(query, sorted_labels, i )
		avg_precision[i] = cnc[i]/(i)
		
	all_precision = np.zeros(binary_mat.shape)
	# determine precision
	for p in range(0,len(binary_mat.T)):
		all_precision[:,p] = all_recall[:,p]/(p+1)
	
	precision = all_precision.mean(axis =0)
	
	tot_recall = all_recall.mean(axis = 0)
	mAP = avg_precision.sum()
	
	return binary_mat,all_recall,tot_recall,precision,avg_precision,cnc,mAP
