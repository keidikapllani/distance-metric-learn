# -*- coding: utf-8 -*-
"""
Mixed Metric Neural Network - MMNN

Created on Tue Dec 11

@author: Antonio Enas
"""
import numpy as np
from src.myutils import load_data,dist_mat, pair_idx,sum_cov,dist_mahalanobis,knn_reid,mma
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA  
from sklearn.metrics import accuracy_score,confusion_matrix
from src.class_PersonReId import PersonReId
from sklearn.cluster import KMeans
import sklearn.utils.linear_assignment_ as la
import scipy.spatial.distance as spd
import matplotlib.pyplot as plt
from src.new_utils import mma_proba_rank
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from re_ranking.re_rank import re_ranking


# Load data
folder = 'PR_data/cuhk03_new_protocol_config_labeled.mat'
camId,filelist,gallery_idx,labels,query_idx,train_idx,features = load_data(folder)

# Split set in custom class
strain = PersonReId(train_idx,camId,labels,features,use_index = True)
squery = PersonReId(query_idx,camId,labels,features,use_index = True)
sgallery = PersonReId(gallery_idx,camId,labels,features,use_index = True)

# PCA for Dimensionality reduction
pca = PCA(n_components = 100) #m=300 rec error of 2,3%
strain_reduced = strain
strain_reduced.features = pca.fit_transform(strain_reduced.features)
sgallery_reduced = sgallery
sgallery_reduced.features = pca.transform(sgallery_reduced.features)
squery_reduced = squery
squery_reduced.features = pca.transform(squery_reduced.features)

# Generate positive/negative pairs from training data
pos_id,neg_id = pair_idx(strain)

# Choose the negative pairs randomly
# as many as the positive pairs to avoid class bias
np.random.seed(0)
mask = np.random.choice(range(0,len(neg_id)), len(pos_id), replace=False)
mask = np.sort(mask)
rnd_neg_id =[neg_id[i] for i in mask]

# Create the new feature vectors
Fp,Yp = mma(strain_reduced,pos_id,label = 1, metrics = ['euclidean','cosine','braycurtis','canberra'])
Fn,Yn = mma(strain_reduced,rnd_neg_id,label = 0, metrics = ['euclidean','cosine','braycurtis','canberra'])

# Pairwise distances in training set to highlight the addressed problem
plt.figure()
plt.subplot(1,2,1)
plt.plot(Fp[:,0],Fp[:,1],'.')
plt.plot(Fn[:,0],Fn[:,1],'r.')
plt.xlabel('Euclidean', fontsize = 16)
plt.ylabel('Cosine', fontsize = 16)
plt.subplot(1,2,2)
plt.plot(Fp[:,2],Fp[:,3],'.')
plt.plot(Fn[:,2],Fn[:,3],'r.')
plt.xlabel('Braycurtis', fontsize = 16)
plt.ylabel('Canberra', fontsize = 16)
plt.suptitle('Pairwise distances in the train set', fontsize = 20)

############################ NEURAL NETWORK ###################################

X_train = np.concatenate((Fp,Fn),axis = 0)
Y_train = np.concatenate((Yp,Yn),axis = 0)
#### Simple alternative initialisation
mlp = MLPClassifier(hidden_layer_sizes=(4,), max_iter=1000, alpha=1e-4,
                    solver='adam', verbose=10, tol=1e-3, random_state = 3,
                    learning_rate_init=.001,activation = 'logistic',batch_size =1000,
					early_stopping = False, validation_fraction = 0.1)

parameter_space = {
    'hidden_layer_sizes': [(4,),(100,),(4,8,4),(50,100,50),(10,20,10) ],
    'activation': ['tanh', 'relu','logistic'],
    'solver': ['adam','sgd'],
    'learning_rate_init': [0.1, 0.01, 0.001, 0.0001],
    'learning_rate': ['constant','adaptive'],
	'tol' : [1e-3,1e-4,1e-5],
	'random_state' : 6,
	'batch_size': [200,300,500,1000],
	'early_stopping':[True],
	'validation_fraction':[0.1,0.2,0.3]
}
clf = GridSearchCV(mlp, parameter_space, n_jobs=3, cv=3,verbose = 10)
clf.fit(X_train, Y_train)

#print("Training set score: %f" % clf.score(X_train[:], Y_train[:]))
#print("Test set score: %f" % clf.score(X_train[:len(tstYp)], Y_train[:len(tstYp)]))
#print("Test set score: %f" % clf.score(X_train[len(tstYp):], Y_train[len(tstYp):]))



mlp.fit(X_train, Y_train)
print("Training set score: %f" % mlp.score(X_train[:], Y_train[:]))
print("Test set score: %f" % mlp.score(X_train[:len(Yp)], Y_train[:len(Yp)]))
ytrain_predict = mlp.predict_proba(X_train)

# Plot the histograms of probabilities predicted for the train
plt.plot()
plt.hist(ytrain_predict[:len(Yp),0])
plt.hist(ytrain_predict[len(Yp):,0])
plt.xlabel('Probability of being dissimilar', fontsize = 14)
plt.ylabel('Number of samples per bin', fontsize = 14)
plt.suptitle('MLPN probability estimate of pair being dissimilar', fontsize = 18)


############################## RERANKING ######################################

# Actual testing phase
metrics = ['euclidean','cosine','braycurtis','canberra']

neighbours_labels,distances,indeces = mma_proba_rank(squery_reduced, sgallery_reduced,metrics, mlp)

final_dist = re_ranking(squery.features,sgallery.features,20,6,0.3, MemorySave = False, Minibatch = 2000)
#Measure accuracy
ng, dg = sgallery.features.shape
nq,dq = squery.features.shape
neighbours_list = []
indices_list = []

distance_rerank = distances + final_dist

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


accuracy = np.zeros((500))
# Codice per accuracy
for i in range(0,500):
    acc, yknn = knn_reid(squery, neighbours_list[j], i+1 )
    accuracy[i] = acc
