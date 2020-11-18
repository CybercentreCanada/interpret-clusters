### Interpret-clusters

Interpret-clusters is an experimental repository extending [interpret](https://github.com/interpretml/interpret), used to interpret unsupervised learning models. For each cluster we want to know "what distinguishes this cluster from the others?". We try to answer this question by training a one-vs-all classifier for each cluster and then interpret the classifier.

In many cases, the set of features used to cluster the data will be different from the features used to interpret the cluster. For example, if you first do dimension reduction (using something like UMAP or t-SNE) on a set of features, the embedded dimensions are not interpretable. You may then try to interpret the clusters on the raw features to gain insight into your results.

  

Interpret-Clusters est un projet expérimental pour étendre les fonctionnalités de [interpret](https://github.com/interpretml/interpret), projet étant utilisé pour interpréter les modèles non supervisés en apprentissage machine. Pour chaque cluster, nous cherchons à savoir « qu’est-ce qui distingue ce cluster d’un autre? ». Nous essayons de répondre à cette question en entrainant un classifieur one-vs-all pour chaque cluster, puis nous interprétons ce classifieur.

Dans plusieurs cas, l'ensemble des caractéristiques utilisées pour cluster les données sera différent des caractéristiques utilisées pour interpréter le cluster. Par exemple, si vous effectuez d'abord une réduction de dimension (en utilisant quelque chose comme UMAP ou t-SNE) sur un ensemble de caractéristiques, les dimensions intégrées ne sont pas interprétables. Vous pouvez alors essayer d'interpréter les clusters à l’aide de leurs caractéristiques brutes pour mieux interpréter vos résultats. 


