import numpy as np

from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors

from trigger_models import TriggerModel

class EDSC(TriggerModel):

    def __init__(self,
                 min_length,
                 max_length,
                 threshold_learning='che',
                 prob_trheshold=.95,
                 bound_threshold=3,
                 alpha=3,
                 min_coverage=1.,
                 n_jobs=1):
        
        super().__init__()

        ######Constant attributes#######
        self.require_classifiers = False
        ################################

        self.min_length = min_length
        self.max_length = max_length
        self.threshold_learning = threshold_learning
        self.prob_threshold = prob_trheshold
        self.bound_threshold = bound_threshold
        self.alpha = alpha
        self.min_coverage = min_coverage

        self.n_jobs = n_jobs

    def _get_bmd(self, X, length):

        sub, bmd = [], []
        for ts in X:
            sub.append(
                [ts[i:length+i] for i in range(len(ts)-length)]
            )
        sub = np.array(sub)

        for i in range(len(X)):
            bmd_f = []
            for shapelet in sub[i]:
                bmd_f.append(
                    self._compute_dist(sub, shapelet)
                )
            bmd.append(bmd_f)

        return np.array(bmd).squeeze(), sub

    def _compute_dist(self, X, shapelet):

        bmd = []
        dists = [[np.linalg.norm(shapelet - candidate) 
                    for candidate in xx] for xx in X]
        bmd.append(
            np.array(dists).min(axis=-1)
        )

        return bmd
     
    def _kernel_density(self, X, x, bandwidth):

        cst = 1/(np.sqrt(2*np.pi) * bandwidth)
        dens = np.exp(-(X - x)**2 / (2*bandwidth)**2).mean()

        return cst * dens 
    
    def _kde_threshold(self, bmd_list, y, idx):

        target = int(y[idx])
        prior_target = (y == target).mean()

        bmd_target = bmd_list[(y == target)]
        bmd_non_target = bmd_list[(y != target)]
        bmd_sort = np.sort(bmd_list)

        h_target = 1.06 * np.std(bmd_target) * (len(bmd_target)**(-0.2))
        h_non_target = 1.06 * np.std(bmd_non_target) * (len(bmd_non_target)**(-0.2))

        dens_target = -self._kernel_density(bmd_target, bmd_sort[-1]/len(y), h_target)
        dens_non_target = -self._kernel_density(bmd_non_target, bmd_sort[-1]/len(y), h_non_target)

        proba = self._compute_proba(prior_target, dens_target, dens_non_target)
        
        threshold = -1
        if proba > self.prob_threshold:
            dens_target = [self._kernel_density(bmd_target, bmd_sort[i], h_target)
                           for i in range(len(bmd_sort))]
            dens_non_target = [self._kernel_density(bmd_non_target, bmd_sort[i], h_non_target)
                               for i in range(len(bmd_sort))]
            probas = np.array(
                [self._compute_proba(prior_target, dens_target[i], dens_non_target[i])
                 for i in range(len(bmd_sort))]
            )
            break_idx = np.where(probas >= self.prob_threshold)[0]
            for idx in range(len(bmd_sort)):
                if idx not in break_idx:
                    break_idx = idx
                    break 

            if break_idx > 1:
                val_candidates = np.linspace(bmd_sort[break_idx-1], bmd_sort[break_idx], 20)
                dens_target = [self._kernel_density(bmd_target, val_candidates[i], h_target)
                               for i in range(len(val_candidates))]
                dens_non_target = [self._kernel_density(bmd_non_target, val_candidates[i], h_non_target)
                                   for i in range(len(val_candidates))]
                probas = np.array(
                    [self._compute_proba(prior_target, dens_target[i], dens_non_target[i])
                     for i in range(len(val_candidates))]
                )
                threshold = val_candidates[
                    np.where(probas >= self.prob_threshold)[0][-1]
                ]

        return threshold, target
        
    def _compute_proba(self, prior_target, dens_target, dens_non_target):

        proba = (prior_target * dens_target) / \
            (((1-prior_target) * dens_non_target) + (prior_target * dens_target))
        
        return proba

    def _che_threshold(self, bmd_list, y, idx):

        target = int(y[idx])
        bmd_non_target = bmd_list[(y != target)]

        threshold = np.mean(bmd_non_target) - self.bound_threshold * np.std(bmd_non_target)

        return threshold, target
    
    def _get_utility(self, X, y, shapelet):

        bmd_list, eml_list = [], []
        for ts in X:
            dists = [np.linalg.norm(shapelet[0] - ts[i:len(shapelet[0])+i])
                     for i in range(len(ts) - len(shapelet[0]))]
            
            bmd_list.append(np.min(dists))
        
            eml = np.where(np.array(dists) <= shapelet[1])[0]
            eml = eml[0] if len(eml) > 0 else np.inf
            eml_list.append(eml + len(shapelet[0]))

        class_mask = (y == shapelet[-1])
        wrecall = np.power(eml_list, (-1/self.alpha))[class_mask].mean()
        
        thresh_mask = (np.array(bmd_list) <= shapelet[1])
        precision = (y[thresh_mask] == shapelet[-1]).mean()

        utility = 2 * precision * wrecall / (precision + wrecall)
        coverage = thresh_mask & (y == shapelet[-1])

        return utility, coverage

    def _learn_shapelets(self, X, y, bmd, length):
        
        shapelets = []
        for i in range(len(X)):
            for j, bmd_list in enumerate(bmd[length][i]):

                if not isinstance(bmd_list, np.ndarray):
                    print("mdr")

                if self.threshold_learning == 'kde':
                    p = self._kde_threshold(bmd_list, y, i)
                elif self.threshold_learning == 'che':
                    p = self._che_threshold(bmd_list, y, i)
                    
                if p[0] > 0:
                    feature = (self.subsequences[length][i,j,:], p[0], p[1])
                    utility, coverage = self._get_utility(X, y, feature)
                    shapelets.append((feature, coverage, utility))

        return shapelets

    def fit(self, X, y):
        
        self.subsequences = []
        results = Parallel(n_jobs=self.n_jobs) \
            (delayed(self._get_bmd)(X, l) 
             for l in range(self.min_length, self.max_length+1))
        
        bmd, self.subsequences = list(map(list, zip(*results)))

        # learn some distance threshold 
        self.shapelets = Parallel(n_jobs=self.n_jobs) \
            (delayed(self._learn_shapelets)(X, y, bmd, l) 
             for l in range(len(bmd)))
        # flatten the list of list
        self.shapelets = sum(self.shapelets, [])
        
        # sort shapelets by utility
        self.shapelets = sorted(
            self.shapelets, key=lambda x: x[-1], reverse=True
        )
        self.features = []
        for shapelet in self.shapelets:

            if len(self.features) == 0:
                self.features.append(shapelet[0])
                current_cov = shapelet[1]
            elif (current_cov | shapelet[1]).sum() > current_cov.sum():
                self.features.append(shapelet[0])
                current_cov = current_cov | shapelet[1]
            
            if current_cov.mean() >= self.min_coverage:
                break

        return self
    
    def predict(self, X):

        predictions, triggers = [], []
        for ts in X:
            # test features by utility ?
            for j, fts in enumerate(self.features):
                sub = fts[0]
                if (len(sub) > len(ts)) and j != len(self.features)-1:
                    continue
                elif (len(sub) > len(ts)) and j == len(self.features)-1:
                    triggers.append(False)
                    predictions.append(np.nan)
                    break

                dists = [np.linalg.norm(sub - ts[i:i+len(sub)])
                         for i in range(len(ts)-len(sub)+1)]
                
                trigger = (np.array(dists) <= fts[1])
                if trigger.any():
                    triggers.append(True)
                    predictions.append(fts[-1])
                    break

                if j == len(self.features)-1:
                    triggers.append(False)
                    predictions.append(np.nan)

        return np.array(predictions), np.array(triggers), None
    

class ECTS(TriggerModel):

    """
    Early classification on time series(2012)
    Inspired by : https://github.com/Eukla/ETS/blob/master/ets/algorithms/ects.py
    """

    def __init__(self, 
                 timestamps, 
                 support=0, 
                 relaxed=False,
                 n_jobs=1):
        """
        Creates an ECTS instance.

        :param timestamps: a list of timestamps for early predictions
        :param support: minimum support threshold
        :param relaxed: whether we use the Relaxed version or the normal
        """
        
        super().__init__()
        
        ######Constant attributes#######
        self.require_classifiers = False
        ################################

        self.rnn = dict()
        self.nn = dict()
        self.mpl = dict()
        self.timestamps = timestamps
        self.support = support
        self.clusters = dict()
        self.occur = dict()
        self.relaxed = relaxed
        self.correct = None

        self.n_jobs = n_jobs

    def fit(self, X, y):

        """
        Function that trains the model using Agglomerating Hierarchical clustering

        :param train_data: a Dataframe containing-series
        :param labels: a Sequence containing the labels of the data
        """
        self.data = X

        self.labels = y
        if self.relaxed:
            self.__leave_one_out()

        indexes, values = np.unique(self.labels, return_counts=True)
        for i, index in enumerate(indexes):
            self.occur[index] = values[i]

        # Finding the RNN of each item
        time_pos = 0
        for e in self.timestamps:
            product = self.__nn_non_cluster(e)  # Changed to timestamps position
            self.rnn[e] = product[1]
            self.nn[e] = product[0]
            time_pos += 1

        temp = {}
        finished = {}  # Dictionaries that signifies if an mpl has been found
        for e in reversed(self.timestamps):
            for index, _ in enumerate(self.data):

                if index not in temp:
                    self.mpl[index] = e
                    finished[index] = 0  # Still MPL is not found
                else:
                    if finished[index] == 1:  # MPL has been calculated for this time-series so nothing to do here
                        continue

                    if self.rnn[e][index] is not None:
                        self.rnn[e][index].sort()
                    # Sorting it in order to establish that the RNN is in the same order as the value
                    if temp[index] is not None:
                        temp[index].sort()

                    if self.rnn[e][index] == temp[index]:  # Still going back the timestamps
                        self.mpl[index] = e
                    else:  # Found k-1
                        finished[index] = 1  # MPL has been found!

                temp[index] = self.rnn[e][index]

        self.__mpl_clustering()
        
        return self

    def __leave_one_out(self):
        nn = []
        for index, row in enumerate(self.data):  # Comparing each time-series

            data_copy = self.data.copy()
            data_copy = np.delete(data_copy, index, axis=0)

            for index2, row2 in enumerate(data_copy):

                temp_dist = np.linalg.norm(row - row2)

                if not nn:
                    nn = [(self.labels[index2], temp_dist)]
                elif temp_dist >= nn[0][1]:
                    nn = [(self.labels[index2], temp_dist)]

            if nn[0][0] == self.labels[index]:
                if not self.correct:
                    self.correct = [index]
                else:
                    self.correct.append(index)
            nn.clear()

    def __nn_non_cluster(self, prefix):

        """Finds the NN of each time_series and stores it in a dictionary

        :param prefix: the prefix with which we will conduct the NN

        :return: two dicts holding the NN and RNN"""

        nn = {}
        rnn = {}

        neigh = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(self.data[:, 0:prefix + 1])
        def something(row):
            return neigh.kneighbors([row])

        result_data = np.apply_along_axis(
            something, axis=1, arr=self.data[:, 0:prefix + 1]
        )
        for index, value in enumerate(result_data):
            value = (value[0], value[1])
            if index not in nn:
                nn[index] = []
            if index not in rnn:
                rnn[index] = []
            for item in value[1][0]:
                if item != index:
                    nn[index].append(item)
                    if item not in rnn:
                        rnn[item] = [index]
                    else:
                        rnn[item].append(index)
        
        return nn, rnn

    def __cluster_distance(self, cluster_a, cluster_b):

        """
        Computes the distance between two clusters as the minimum among all
        inter-cluster pair-wise distances.

        :param cluster_a: a cluster
        :param cluster_b: another cluster
        :return: the distance
        """

        min_distance = np.inf
        for i in cluster_a:
            for j in cluster_b:
                d = np.linalg.norm(self.data[i] - self.data[j])
                if min_distance > d:
                    min_distance = d

        return min_distance

    def nn_cluster(self, cl_key, cluster_index):

        """Finds the nearest neighbor to a cluster
        :param cluster_index: List of indexes contained in the list
        :param cl_key: The key of the list in the cluster dictionary
        """
        #global x
        dist = np.inf
        candidate = []  # List that stores multiple candidates

        for key, value in self.clusters.items():  # For each other cluster

            if cl_key == key:  # Making sure its a different to our current cluster
                continue
            temp = self.__cluster_distance(cluster_index, value)  # Find their Distance

            if dist > temp:  # If its smaller than the previous, store it
                dist = temp
                candidate = [key]

            elif dist == temp:  # If its the same, store it as well
                candidate.append(key)
        #x-=1
        return candidate

    def __rnn_cluster(self, e, cluster):

        """
        Calculates the RNN of a cluster for a certain prefix.

        :param e: the prefix for which we want to find the RNN
        :param cluster: the cluster that we want to find the RNN
        """

        rnn = set()
        complete = set()
        for item in cluster:
            rnn.union(self.rnn[e][item])
        for item in rnn:
            if item not in cluster:
                complete.add(item)
        return complete

    def __mpl_calculation(self, cluster):

        """Finds the MPL of discriminative clusters
        
        :param cluster: The cluster of which we want to find it's MPL"""

        # Checking if the support condition is met
        index = self.labels[cluster[0]]
        if self.support > len(cluster) / self.occur[index]:
            return
        mpl_rnn = self.timestamps[len(self.timestamps) - 1]  
        # Initializing the  variables that will indicate the 
        # minimum timestamp from which each rule applies
        mpl_nn = self.timestamps[len(self.timestamps) - 1]
        """Checking the RNN rule for the clusters"""
        
        # Finding the RNN for the L
        curr_rnn = self.__rnn_cluster(self.timestamps[len(self.timestamps) - 1], cluster)

        if self.relaxed:
            curr_rnn = curr_rnn.intersection(self.correct)

        for e in reversed(self.timestamps):

            temp = self.__rnn_cluster(e, cluster)  # Finding the RNN for the next timestamp
            if self.relaxed:
                temp = temp.intersection(self.correct)
            # If their division is an empty set, then the RNN is the same so the
            if not curr_rnn - temp:  
                # MPL is e
                mpl_rnn = e
            else:
                break
            curr_rnn = temp

        """Then we check the 1-NN consistency"""
        rule_broken = 0
        for e in reversed(self.timestamps):  # For each timestamp

            for series in cluster:  # For each time-series

                for my_tuple in self.nn[e][series]:  # We check the corresponding NN to the series
                    if my_tuple not in cluster:
                        rule_broken = 1
                        break
                if rule_broken == 1:
                    break
            if rule_broken == 1:
                break
            else:
                mpl_nn = e
        for series in cluster:
            pos = max(mpl_rnn, mpl_nn)  # The value at which at least one rule is in effect
            if self.mpl[series] > pos:
                self.mpl[series] = pos

    def __mpl_clustering(self):

        """Executes the hierarchical clustering"""
        n = self.data.shape[0]
        redirect = {}  # References an old cluster pair candidate to its new place
        discriminative = 0  # Value that stores the number of discriminative values found
        """Initially make as many clusters as there are items"""
        for index in range(len(self.data)):
            self.clusters[index] = [index]
            redirect[index] = index

        result = []
        """Clustering loop"""
        while n > 1:  # For each item
            closest = {}
            my_list = list(self.clusters.items())
            res = Parallel(n_jobs=self.n_jobs) \
                (delayed(self.nn_cluster)(k, idx) for k, idx in my_list)
            
            for key,p  in zip(self.clusters.keys(),res):
                closest[key] = p

            for key, value in closest.items():
                for item in list(value):
                    if key in closest[item]:  # Mutual pair found
                        closest[item].remove(key)
                        #If 2 time-series are in the same cluster
                        # (in case they had an 3d  neighboor that invited them in the cluster)
                        if  redirect[item]==redirect[key]:  
                            continue
                        for time_series in self.clusters[redirect[item]]:
                            self.clusters[redirect[key]].append(time_series)  # Commence merging
                        del self.clusters[redirect[item]]
                        n = n - 1
                        redirect[item] = redirect[key]  # The item can now be found in another cluster
                        for element in self.clusters[redirect[key]]:  # Checking if cluster is discriminative
                            result.append(self.labels[element])

                        x = np.array(result)
                        if len(np.unique(x)) == 1:  # If the unique class labels is 1, then the
                            # cluster is discriminative
                            discriminative += 1
                            self.__mpl_calculation(self.clusters[redirect[key]])

                        for neighboors_neigboor in closest:  # The items in the cluster that has been assimilated can
                            # be found in the super-cluster
                            if redirect[neighboors_neigboor] == item:
                                redirect[neighboors_neigboor] = key
                        result.clear()
            if discriminative == 0:  # No discriminative clusters found
                break
            discriminative = 0

    def predict(self, X):

        """
        Prediction phase.
        Finds the 1-NN of the test data and if the MPL oof the closest 
        time-series allows the prediction, then return that prediction
         """
        
        predictions, triggers, times_idx = [], [], []
        nn = []
        candidates = []  # will hold the potential predictions
        cand_min_mpl = []

        for test_row in X:
            for e in self.timestamps:
                neigh = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(self.data[:, 0:e+1])
                neighbors = neigh.kneighbors([test_row[0:e+1]])
                candidates.clear()
                cand_min_mpl.clear()
                nn = neighbors[1]
                for i in nn:
                    if e >= self.mpl[i[0]]:
                        candidates.append((self.mpl[i[0]], self.labels[i[0]]))  # Storing candidates by mpl and by label
                if len(candidates) > 1:  # List is not empty so wee found candidates
                    candidates.sort(key=lambda x: x[0])
                    for candidate in candidates:
                        if candidate[0] == candidates[0][0]:
                            cand_min_mpl.append(candidate)  # Keeping the candidates with the minimum mpl
                        else:
                            break  # From here on the mpl is going to get bigger
                    predictions.append(max(set(cand_min_mpl), key=cand_min_mpl.count))  # The second argument is the max label
                    triggers.append(True)
                    times_idx.append(e)
                    break
                elif len(candidates) == 1:  # We don't need to to do the above if we have only one nn
                    predictions.append(candidates[0][1])
                    triggers.append(True)
                    times_idx.append(e)
                    break

            if len(candidates) == 0:
                triggers.append(False)
                predictions.append(np.nan)
                times_idx.append(self.timestamps[-1])

        return np.array(predictions), np.array(triggers), np.array(times_idx)