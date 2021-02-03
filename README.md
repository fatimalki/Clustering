# Clustering short texts

## Overview

This project aims at finding meaningful clusters in short texts. Each short text is read as a separate document, then a set of pre processing steps are done on these short texts. Once, we are done with the NLP tasks, we use the popular machine learning algorithm, K-Means, to cluster these documents into meaningful categories/clusters.

The Clustering.py file takes in input a single document containing short texts.  The Clustering.py will then print the clusters for the short texts in the order of the short text. Also, for every cluster, we are printing two answers that are closest to the centroids and two that are farthest from the centroids. For calculating distance, we use euclidean distance between the two vectors.

## Dataset

The data used in the project is the text file "Shorttexts.txt". We assume that each short text is on a single line in the text file. Each short text is considered as a document.

## Data preparation

This is a Natural Language Processing (NLP) project. NLP is a way of processing textual information into numerical summaries, features, or models. 

Data Preparation is a crucial step in NLP projects. Hence, the input data needs to be cleaned and preprocessed and transformed before the modelisation step. The major preparations used on input data are :

1- Convertion of text to lowercase.  
2- Removing return to line.  
3- Replacing all kind of spaces by a character space.  
4- Tokenization : breaking up the strings into a list of words.  
5- Lemmatization (text normalization) : shortening words back to their root form to remove noise in the data.  
6- Transforming a given text into a vector on the basis of the frequency of each word that occurs in the entire text (TF-IDF).

## Clustering algorithm

The project implements the K-Means algorithm for clustering. This algorithm iteratively recomputes the location of k centroids (k is the number of clusters, defined beforehand), that aim to classify the data. Points are labelled to the closest centroid, with each iteration updating the centroids location based on all the points labelled with that value.

## Dependencies

You need Python 3 to run the packages. Other dependencies can be found in the requirement file : Requirements.txt

## Documentation

You can learn more on K-Means at https://datasciencegeeks.net/2016/03/16/understanding-k-means-clustering/

You can know more about the concept of TF-IDF at https://monkeylearn.com/blog/what-is-tf-idf/










