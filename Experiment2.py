#!/usr/local/bin/python3
# Submitted by: Dhruva Bhavsar, Username:dbhavsar
# Programming Project 1: Experiment 2

import numpy as np
import sys
from math import log
import random
import matplotlib.pyplot as plt

# Read data from the file
def read_file(filename):
    dataset=[]
    with open(filename, "r") as f:
        for line in f.read().lower().rstrip().split("\n"):
            a=line.split("\t")
            dataset.append(a)
        return dataset

# Train the program for given dataset/trainset and smoothing factor 'm'  
def train_data(dataset,m):
    wordlist=[]
    positivewords=[]
    negativewords=[]
    posfreq=[]
    negfreq=[]
    count=0
    poscount=0
    negcount=0
    for word in dataset:
        wordlist += word[0].split(" ")
        if(word[1]=='1'):
            positivewords+=word[0].split(" ")
        elif(word[1]=='0'):
            negativewords+=word[0].split(" ")
    # Make a list of unique words
    wordlist=set(wordlist)
    
    # Calculate the total number of words in vocabulary and total words in positive and negative class
    for w in wordlist:
        count+=1
        posfreq.append(positivewords.count(w))
        negfreq.append(negativewords.count(w))
    for w in positivewords:
        poscount+=1
    for w in negativewords:
        negcount+=1
        
    # Calculate MLE or MAP for each word depending on smoothing factor m
    posfreq=[w+m/(poscount+(m*count)) for w in posfreq]
    negfreq=[w+m/(negcount+(m*count)) for w in negfreq]
    trainset=list(zip(wordlist,posfreq,negfreq))
    return trainset   
 
# Split a dataset into k folds
# This function block has been taken from https://machinelearningmastery.com/implement-resampling-methods-scratch-python/
def cross_validation_split(dataset, folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / folds)
	for i in range(folds):
		fold = list()
		while len(fold) < fold_size:
			index = random.randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split
# ------END--------

def test_data(testset,trainset):
    pred=[]
    for sen in testset:
        words=sen[0].split(" ")
        posprob=0
        negprob=0
        for w in words:
            for data in trainset:
                # If word in testset matches any word in trainset, calculate + and - probability using MLE/MAP table
                if(w==data[0]):
                    if(data[1]>0):
                        posprob+=log(data[1])
                    if(data[2]>0):
                        negprob+=log(data[2])
        # Check if sentence is + or - based on which probability is more. If both are 0, predict any random value
        if(posprob>negprob):
            pred.append((sen[0],sen[1],'1'))
        elif(negprob>posprob):
            pred.append((sen[0],sen[1],'0'))
        else:
            r=random.randint(0,1)
            pred.append((sen[0],sen[1],str(r)))
       
    return pred        

if __name__ == "__main__":
    # Read data from the given dataset files having title like '~labelled.txt'
   data=read_file(sys.argv[1])
   
   step=float(sys.argv[2])
   
   split=cross_validation_split(data,10)
   
   sf=[]
   m1=[]
   s1=[]
   if step==1:
       m=1
   elif step==0.1:
       m=0
   
   for l in range(0,10):
       acc=np.empty((10,10))
       size=np.zeros((10,1))
       acc1=[]
       for j in range(0,10):
           count=0
           trainset=[]
           # Separate trainset and testset
           for i in range(0,10):
               if(i!=j):
                   trainset.extend(train_data(split[i],m))
           N=len(trainset)
           train=[]
           train_i=[]
          
           # Predicting result using ith split as test data
           pred=test_data(split[j],trainset)
           total=len(pred)
           count=0
                        
           # Calculate the accuracy for each sample
                        
           for l in pred:
                if l[1]==l[2]:
                    count+=1
           accuracy=(count/total)*100
           acc1.append(accuracy)
         
       mean=[]
       std=[]
                
       #Calculate mean and standard deviation for each sample size
       
       mean=np.mean(acc1)
       std=np.std(acc1)
                
       m1.append(mean)
       s1.append(std)
       sf.append(m)       
       
       m+=step #Increment m    
       
   # Plot the graph of accuracy vs. smoothing factor m
   plt.xlabel('Smoothing factor')
   plt.ylabel('Accuracy')
#            line1=plt.plot(sf,m1)
   plt.errorbar(sf,m1,yerr=s1)
   plt.show()
