import logging
from copy import deepcopy
from warnings import warn

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from interpret.glassbox import ExplainableBoostingClassifier, LogisticRegression
from tqdm import tqdm


class ClusterExplainer():
    
    def __init__(self, features, cluster_labels, feature_names=None, clusters_to_analyze=None, 
                 classifier='ebm', score_threshold=0.8, verbose=False):
        
        """Interpret-clusters is a utility that aims to provide cluster interpretations. This is done by using the cluster ids as labels 
        and training supervised learning models to predict the clusters. The given features do not need to be the same set of features 
        as what was used to calculate the clusters. By calculating the feature importance of the supervised model we can find the features 
        that are important to distinguishing a particular cluster. 
        
        Parameters
        ----------

        features: array or pandas.DataFrame
            The set of features to pass to the supervised learning model. This does not need to be the same set of features as what was used to calculate the clusters.
        
        cluster_labels: list
            The list of cluster labels that specify the cluster to which a point belongs. This must have the same dimension as features (i.e. there must be one label per data point).

        feature_names: list (optional, default None)
            The list of feature names which correspond to the columns of features. If None the column indices will be used.
        
        clusters_to_analyze: list (optional, None)
            The list of cluster labels to calculate feature importances for. If None then all clusters will be analyzed.
        
        classifier: string or callable (optional, default ebm)
            The classifier to use for predicting cluster labels. It must be a classifier from the interpret package. Built-in options are ["ebm", "logistic_regression"].

        score_threshold: float (optional, default 0.8)
            Warn if the trained model has a score below this threshold.
        
        verbose: bool (optional, default False)
            Display progress information.

        """
        self.features = features
        self.cluster_labels = np.array(cluster_labels)
        
        if feature_names is not None:
            self.feature_names = np.array(feature_names)
        else:
            self.feature_names = np.arange(features.shape[1])
        
        self.cluster_models = {}
        
        if clusters_to_analyze is None:
            self.clusters_to_analyze = list(set(self.cluster_labels))
        else:
            self.clusters_to_analyze = sorted(clusters_to_analyze)
        
        for cluster_id in self.clusters_to_analyze:
            if classifier == 'ebm':
                classifier = ExplainableBoostingClassifier(feature_names=self.feature_names)
            elif classifier == 'logistic_regression':
                classifier = LogisticRegression(feature_names=self.feature_names, penalty='l1', solver='liblinear')

            cluster_model = ClusterModel(cluster_id, deepcopy(classifier), features, cluster_labels, score_threshold=score_threshold, verbose=verbose)
            self.cluster_models[cluster_id] = cluster_model
    
        self.verbose = verbose
        self.local_explanations = {}
        self.global_explanations = {}

    def cluster_local_explanations(self, cluster_label):
        """Find the important features for all points in a specific cluster

        Parameters
        ----------
        cluster_label: str or int
            The label of the cluster to analyze. 
        """
        clf_local = self.local_explanations.get(cluster_label, None)

        if clf_local is None:
            cluster_model = self.cluster_models[cluster_label]
            cluster_model.fit(self.features)
            clf_local = self.local_explanations[cluster_label] = cluster_model.explain_local(self.features)
        
        return clf_local

    def calculate_all_local_explanations(self):
        """Calculate local explanations for all clusters in clusters_to_analyze."""
        for cluster_label in (tqdm(self.clusters_to_analyze) if self.verbose else self.clusters_to_analyze):
            clf_local = self.cluster_local_explanations(cluster_label)
            self.local_explanations[cluster_label] = clf_local

    def cluster_global_explanations(self, cluster_label):
        """Find the important features for a specific cluster

        Parameters
        ----------
        cluster_label: str or int
            The label of the cluster to analyze. 
        """
        cluster_model = self.cluster_models[cluster_label]
        cluster_model.fit(self.features)
        clf_global = self.global_explanations[cluster_label] = cluster_model.explain_global()
        return clf_global

    def calculate_all_global_explanations(self):
        """Calculate global explanations for all clusters in clusters_to_analyze."""
        for cluster_label in (tqdm(self.clusters_to_analyze) if self.verbose else self.clusters_to_analyze):
            clf_global = self.cluster_global_explanations(cluster_label)
            self.global_explanations[cluster_label] = clf_global

    def get_model_score_for_cluster(self, cluster_label):
        """Get a performance score for the ClusterModel
        
        Parameters
        ----------
        cluster_label: str or int
            The label of the cluster to analyze.         
        """
        if cluster_label in self.clusters_to_analyze:
            model_score = self.cluster_models[cluster_label].score
            if model_score is not None:
                return model_score
            else:
                self.cluster_models[cluster_label].fit(self.features)
                return self.cluster_models[cluster_label].score
        else:
            raise KeyError('Could not find cluster_label {} in list of clusters to analyze. It must be found in {}'.format(cluster_label, self.clusters_to_analyze))      


class ClusterModel():
    
    def __init__(self, label, classifier, features, cluster_labels, score_threshold=0.8, verbose=False):
        """A supervised learning model trained to separate the cluster from all other data points.

        Parameters
        ----------
        label: str or int
            The label of the cluster to analyze.
        
        classifier: callable
            The classifier to use for predicting cluster labels. It must be a tree based classifier from sklearn.
        
        score_threshold: float (optional, default 0.8)
            Warn if the trained model has a score below this threshold.
        
        verbose: bool (optional, default False)
            Display progress information.
        """
        self.label = label
        self.model = classifier
        self.ranked_features = None
        self.score = None
        self.verbose = verbose
        self.name = f'Cluster {self.label}'
        self.score_threshold = score_threshold

        self.get_cross_validation_set_indices(features, cluster_labels)

    def get_cross_validation_set_indices(self, features, labels):
        """Split the features into training and test sets and return the indices."""
        # Classifier will be one vs all
        one_vs_all_labels = [1 if self.label == label else 0 for label in labels]
        self.one_vs_all_labels = np.array(one_vs_all_labels)
        indices = list(range(len(self.one_vs_all_labels)))
        _, _, _, _, self.train_indices, self.test_indices = train_test_split(features, self.one_vs_all_labels, indices, test_size=0.3)
    
    def fit(self, features):
        """Train a supervised learning model to separate the points in the given cluster from all other points."""
        # Train a 1 vs all classifier
        if self.score is None:
            if self.verbose:
                print(f'Training model for cluster {self.label}')
            
            self.model.fit(features[self.train_indices], self.one_vs_all_labels[self.train_indices])
            self.score = self.model.score(features[self.test_indices], self.one_vs_all_labels[self.test_indices])

            if self.verbose:
                print(f'Finished training model for cluser {self.label}. Score for model is {self.score}')

            if self.score < self.score_threshold:
                warn(f'Model score for cluster {self.label} is {self.score} which is below the threshold of {self.score_threshold}')

    def explain_local(self, features):
        """Calculate explanations for each individual point"""
        is_cluster = self.one_vs_all_labels == 1
        cluster_of_interest_features = features[is_cluster]
        cluster_of_interest_labels = self.one_vs_all_labels[is_cluster]

        if self.verbose:
            print(f'Calculating local explanations for cluster {self.label}')

        local_explanations = self.model.explain_local(cluster_of_interest_features, cluster_of_interest_labels, name=self.name)
        return local_explanations

    def explain_global(self):
        """Calculate explanations for cluster as a whole and for each feature overall."""
        if self.verbose:
            print(f'Calculating global explanations for cluster {self.label}')

        global_explanations = self.model.explain_global(name=self.name)
        return global_explanations






