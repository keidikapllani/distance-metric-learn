# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 22:43:09 2018

@author: AE KK
"""
import numpy as np
import scipy.spatial.distance as spd
from collections import defaultdict

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

def secret_idx(query,gallery):
	'''
	Generate two index lists, a list of positive (same identity but different
	camera) and a list of negative pairs (different identity) to facilitate
	metric learning with KISSME.
	'''
	#initialise output variables
	negative_id = []
	positive_id = []
	# for each data point
	for n in range(0,len(query.idx)):
		# Print completion status
		c = 100*n/query.n
		print(f'{c}% Completed')
		
		idx1 = query.idx[n]
		# combine with all other pictures but itself
		for i in range(0,len(gallery.idx)):	
			
			# if not same identity match the pair in the negatives
			if query.labels[n] != gallery.labels[i]:
				negative_id.append((idx1,gallery.idx[i]))
			
			# else if same label and different cam append in the positives
			elif query.labels[n] == gallery.labels[i] and query.camId[n] != gallery.camId[i]:				
				positive_id.append((idx1,gallery.idx[i]))
			
			# else ignore
	return positive_id, negative_id


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

	return all_cmc, mAP,all_AP

