import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from interpret.glassbox import ExplainableBoostingClassifier
from tqdm import tqdm

from copy import deepcopy

class ClusterExplainer():
    
    def __init__(self, features, cluster_labels, feature_names=None, clusters_to_analyze=None, 
                 classifier='ebm', include_training_set=False):
        
        """Looking Glass is a utility that aims to provide cluster interpretations. This is done by using the cluster ids as labels and training supervised learning models to predict the clusters. 
        The given features do not need to be the same set of features as what was used to calculate the clusters. By calculating the feature importance of the supervised model (using SHAP values) 
        we can find the features that are important to distinguishing a particular cluster. 
        
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
            The classifier to use for predicting cluster labels. It must be a classifier from the interpret package.

        include_training_set: bool (optional, False)
            Whether or not to include the training set when calculating feature importances. By default only the test set is used.
        """
        self.features = features
        self.cluster_labels = np.array(cluster_labels)
        self.include_training_set = include_training_set
        
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
            
            cluster_model = ClusterModel(cluster_id, deepcopy(classifier), features, cluster_labels)
            self.cluster_models[cluster_id] = cluster_model
    
        self.local_explanations = {}

    def calculate_feature_ranking_for_cluster(self, cluster_label):
        """Find the important features for a specific cluster

        Parameters
        ----------
        cluster_label: str or int
            The label of the cluster to analyze. 
        """
        cluster_model = self.cluster_models[cluster_label]
        cluster_model.fit(self.features)
        clf_local = self.local_explanations[cluster_label] = cluster_model.explain_local(self.features)
        return clf_local
        # cluster_model.calculate_shap_scores(self.features, include_training_set=self.include_training_set)
        # cluster_model.calculate_feature_rankings(self.feature_names)
        
    def calculate_feature_rankings(self):
        """Calculate feature importances for all clusters in clusters_to_analyze."""
        for cluster_label in tqdm(self.clusters_to_analyze):
            clf_local = self.calculate_feature_ranking_for_cluster(cluster_label)
            # cluster_model = self.cluster_models[cluster_label]
            self.local_explanations[cluster_label] = clf_local
        
        # return self.local_explanations

    # def get_ranking_for_cluster(self, cluster_label):
    #     """Get a ranked list of feature importances for a given cluster.
        
    #     Parameters
    #     ----------
    #     cluster_label: str or int
    #         The label of the cluster to analyze.         
    #     """
    #     if cluster_label in self.clusters_to_analyze:
    #         ranking = self.cluster_models[cluster_label].ranked_features
    #         if ranking is not None:
    #             return ranking
    #         else:
    #             self.calculate_feature_ranking_for_cluster(cluster_label)
    #             return self.cluster_models[cluster_label].ranked_features
    #     else:
    #         raise KeyError('Could not find cluster_label {} in list of clusters to analyze. It must be found in {}'.format(cluster_label, self.clusters_to_analyze))

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
    
    def __init__(self, label, classifier, features, cluster_labels):
        """A supervised learning model trained to separate the cluster from all other data points.

        Parameters
        ----------
        label: str or int
            The label of the cluster to analyze.
        classifier: callable
            The classifier to use for predicting cluster labels. It must be a tree based classifier from sklearn.
        """
        self.label = label
        self.model = classifier
        self.ranked_features = None
        self.score = None

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
        # self.get_cross_validation_set_indices(features, labels)
        # Train a 1 vs all classifier
        self.model.fit(features[self.train_indices], self.one_vs_all_labels[self.train_indices])
        self.score = self.model.score(features[self.test_indices], self.one_vs_all_labels[self.test_indices])

    def explain_local(self, features):
        is_cluster = self.one_vs_all_labels == 1
        cluster_of_interest_features = features[is_cluster]
        cluster_of_interest_labels = self.one_vs_all_labels[is_cluster]

        ebm_local = self.model.explain_local(cluster_of_interest_features, cluster_of_interest_labels)
        
        return ebm_local

    # def calculate_shap_scores(self, features, include_training_set=False):
    #     """Calculate SHAP values for each of the features.
        
    #     Parameters
    #     ----------
    #     features: array or pandas.DataFrame
    #         The set of features used to train the supervised learning model.
    #     include_training_set: bool (optional, default False)
    #         Whether or not to include the training set when calculating feature importances. By default only the test set is used.
    #     """
    #     self.explainer = shap.TreeExplainer(self.model)
        
    #     if include_training_set:
    #         shap_values = self.explainer.shap_values(features)
    #     else:
    #         shap_values = self.explainer.shap_values(features[self.test_indices])
    
    #     self.shap_values = shap_values

    # def calculate_feature_rankings(self, feature_names):
    #     """Rank the features by their mean SHAP score.

    #     Parameters
    #     ----------
    #     feature_names: list
    #         The list of feature names. If none are provided then the indices will be used.
        
    #     """
    #     # The 1 is due to the class label being 1
    #     cluster_shap_values = self.shap_values[1]
    #     # Calculate the mean SHAP value for each feature
    #     mean_shap_values = cluster_shap_values.mean(axis=0)
    #     sorted_means = np.argsort(mean_shap_values, axis=0)
    #     self.ranked_features = [(feature_names[idx], mean_shap_values[idx]) for idx in reversed(sorted_means)]






