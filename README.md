# Programming Project 1:

The code is divided into two parts:

1)Experiment1.py : This program checks for all files in current folder having name ending with "labelled.txt".
For each dataset file, it generates a graph plotting the accuracies and standard deviation against the sampling size for both m=0 and m=1.

Input: 
```
 python Experiment1.py
```

2)Experiment2.py : This program takes as input the full name of file with extension and the factor by which to increase the value of m. 
For eg: For the series of m=0, 0.1, 0.2, …, 0.9 the value of step should be given as 0.1 . Similarly, for m=1,2,…,10 value of step should be 1.

Input:
```
 python Experiment2.py "yelp\_labelled.txt" 1
````