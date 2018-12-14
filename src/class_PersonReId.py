# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 18:29:11 2018

@author: AE
"""
import numpy as np

class PersonReId:
	'''
	Object class for person re-id datasets.
	'''
	# Self attributes
	def __init__(self,idx,camId,labels,features,use_index = False):
		if use_index == True:
			self.idx = idx
			self.camId = camId[idx]
			self.labels = labels[idx]
			self.features = features[idx]
			self.d = len(features[idx].T) #dimensionality D
			self.n = len(idx)			  #size of dataset N
			self.c = np.unique(self.labels) #number of classes
		else:
			self.idx = idx
			self.camId = camId
			self.labels = labels
			self.features = features
			self.d = len(features.T) #dimensionality D
			self.n = len(idx)		 #size of dataset N
			self.c = np.unique(self.labels)
		        
	# Class methods
	def camId_of(self, index):
		'''
		Returns the camera ID for the specific input index
		'''		 
		cam = self.camId[self.idx == index]
		return cam
	
	def label_of(self, index):
		'''
		Returns the person ID for the specific input index
		'''		 
		pr = self.labels[self.idx == index]
		return pr
	
	def features_of(self, index):
		'''
		Returns the feature vector for the specific input index
		'''		 
		ft = self.features[self.idx == index]
		return ft
		
	def removal_of(self,query_idx):
		'''
		Remove instancies of the query person taken from the same camera from
		the dataset.
		'''
		#Exclude if they have same ID and same camera
		keep_indexes = ~((self.camId == self.camId_of(query_idx)) & (self.labels == self.label_of(query_idx)))
		dataset = PersonReId(
				self.idx[keep_indexes],
				self.camId[keep_indexes],
				self.labels[keep_indexes],
				self.features[keep_indexes])
		return dataset
	
	def ignore_cam(self,cameraId):
		'''
		Ignore pictures taken from a camera.
		'''
		keep_indexes = self.camId != cameraId
		dataset = PersonReId(
				self.idx[keep_indexes],
				self.camId[keep_indexes],
				self.labels[keep_indexes],
				self.features[keep_indexes])
		return dataset

	def exclude_cam_label(self,query_cam,query_label):
		'''
		Remove instancies of the query person taken from the same camera from
		the dataset.
		'''
		#Exclude if they have same ID and same camera
		keep_indexes = ~((self.camId == query_cam) & (self.labels == query_label))
		dataset = PersonReId(
				self.idx[keep_indexes],
				self.camId[keep_indexes],
				self.labels[keep_indexes],
				self.features[keep_indexes])
		return dataset
	
	def query_gallery_split(self):
		'''
		Split the dataset into two
		'''
		query_bool = np.zeros((self.n,),bool)
		#for each 
		cnt = 0
		taken1 = False
		taken2 = False
		for i in range(0,self.n):
			if self.camId[i] == 1 and self.labels[i] == self.c[cnt] and taken1 == False:
				query_bool[i] = 1
				taken1 = True
				taken2 = False
			elif self.camId[i] == 2 and self.labels[i] == self.c[cnt] and taken2 == False:
				query_bool[i] = 1
				cnt +=1
				taken1 = False
				taken2 = True
			else:
				query_bool[i] = 0
			
		gallery_bool = np.invert(query_bool)
		queryset = PersonReId(
				self.idx[query_bool],
				self.camId[query_bool],
				self.labels[query_bool],
				self.features[query_bool])
		
		galleryset = PersonReId(
				self.idx[gallery_bool],
				self.camId[gallery_bool],
				self.labels[gallery_bool],
				self.features[gallery_bool])
		
		return queryset,galleryset