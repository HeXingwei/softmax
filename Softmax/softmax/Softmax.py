#-*- coding=utf-8 -*-
__author__ = 'hxw'
import numpy as np
class Softmax(object):
	"class softmax"
	def __init__(self,size,epo=1000,rate=0.001,lamda=0.1):
		"""
		:param size: size=(num of attributes,num of classes)
		:param epo: training
		:param rate:learning rate
		:return:
		"""
		self.epo=epo
		self.rate=rate
		self.lamda=lamda
		self.weights=np.random.normal(size=size)
	def fit(self,traindata,testdata=None):
		"train softmax and use stochastic gradient decent updating weights"
		best_accuracy=-1
		best_epo=-1
		for i in range(self.epo):
			print "epo %d"%i
			for j in range(data.shape[0]):
				x=data[j,:-1]
				y=data[j,-1]
				h=self.softmax(x)
				#print h
				#update weights
				self.weights=self.weights.transpose()
				for k in range(self.weights.shape[0]):
					if k==y:
						self.weights[k]+=self.rate*(1-h[k])*x-self.lamda*self.weights[k]
					else:
						self.weights[k]+=self.rate*(-h[k])*x-self.lamda*self.weights[k]
				self.weights=self.weights.transpose()
			if traindata is not None:
				accu=self.accuracy(traindata)
				if accu>best_accuracy:
					best_epo=i
					best_accuracy=accu
			print "best_epo is %d ,best_accuracy is %lf"%(best_epo,best_accuracy)
	def softmax(self,x):
		h=np.exp(self.weights.transpose().dot(x))/np.sum(np.exp(self.weights.transpose().dot(x)))
		h=np.nan_to_num(h)
		return h
	def predict(self,x):
		h=np.argmax(self.softmax(x))
		return h
	def accuracy(self,data):
		num=0
		for i in range(data.shape[0]):
			x=data[i][:-1]
			y=data[i][-1]
			h=self.predict(x)
			if y==h:
				num+=1
		#print "predict accuracy is %lf"%(num*1.0/data.shape[0])
		return num*1.0/data.shape[0]
def loadData(path):
	data=np.loadtxt(path,skiprows=21,dtype="int32",delimiter=",")
	new_data=np.ones((data.shape[0],data.shape[1]+1))
	""""relarge the dataset with x0=1"""
	new_data[:,1:]=data
	return new_data
data=loadData("D:\\SelfLearning\\Machine Learning\\ClassifyDataSet\\penbased\\penbased.dat")
train_data=data[:8000]
test_data=data[8000:]
lr=Softmax(size=(17,10),epo=150,lamda=0.00001)
lr.fit(train_data,test_data)

print lr.accuracy(test_data)
print lr.accuracy(train_data)