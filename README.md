# Clustering short texts
___________________________________________

## Overview

This project aims at finding meaningful clusters in short texts. Each short text is read as a separate document, then a set of pre processing steps are done on these short texts. Once, we are done with the NLP tasks, we use the popular machine learning algorithm, K-Means, to cluster these documents into meaningful categories/clusters.

The Clustering.py file takes in input a single document containing short texts.  The Clustering.py will then print the clusters for the short texts in the order of the short text. Also, for every cluster, we are printing two answers that are closest to the centroids and two that are farthest from the centroids. For calculating distance, we use euclidean distance between the two vectors.

## Dataset

The data used in the project is the text file "Shorttexts.txt". We assume that each short text is on a single line in the text file. Each short text is considered as a document.

## Data preparation

## Natural Language Processing (NLP)

Natural Language Processing (NLP) is a way of processing textual information into numerical summaries, features, or models.

## Clustering algorithm

The project implements the K-Means algorithm for clustering. This algorithm iteratively recomputes the location of k centroids (k is the number of clusters, defined beforehand), that aim to classify the data. Points are labelled to the closest centroid, with each iteration updating the centroids location based on all the points labelled with that value.

## Documentation

You can learn more on K-Means at https://datasciencegeeks.net/2016/03/16/understanding-k-means-clustering/

## Results

## Dependencies










