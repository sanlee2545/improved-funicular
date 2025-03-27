"""Nearest neighbors predictor."""

from ball_tree import Ball_Tree, Node
from functions import (
    bin_and_one_hot_encode, classify, classify_accuracy, distance_L1,
    distance_L2, dot_product, insertion_sort_parallel,
    mean_squared_error, regress
)
from math import ceil, floor, pow, sqrt
from random import choices, random, sample
from statistics import quantiles
from time import perf_counter

class Nearest_Neighbors:
    """Nearest neighbor predictor with dispatch function.
    
    Attributes :
        dataset (2D list) : A copy of the input training dataset
            without the first row; the feature labels.
        feature_labels (1D list) : List of strings for each feature.
            Can be either "categorical" or "numerical".
        cols (int) : Number of columns or features.
        rows (int) : Number of rows or training data points.
        distance_function (method) : One of the following norms : L1,
            L2, or LInfinity(not yet added).
        prediction_function (method) : Function used to compute the
            prediction; can be "regress" or "classify".
        k (int) : Number of near neighbors to look for.
        predictions (1D list) : List of predictions.
        priority_queue (1D list) : Fake priority queue used only for
            ball tree search.
        variant (string) : Type of nearest neighbor predictor to use.
    """

    def __init__(self, dataset, distance_function, k, variant):
        """Initialize Nearest Neighbor predictor.
         
        Args :
            dataset (2D list) : Training dataset. We assume that the
                rows are data points and columns are features.
            distance_function (string) : One of the following norms :
                L1, L2, or LInfinity(not yet added).
            k (int) : Number of near neighbors to look for.
            variant (string) : Type of nearest neighbor predictor to
                use.
        """
        
        self.dataset = dataset[1:]
        self.feature_labels = dataset[:1][0]
        
        self.cols = len(self.dataset[0])
        self.rows = len(self.dataset)

        if distance_function == "L1":
            self.distance_function = distance_L1
        elif distance_function == "L2":
            self.distance_function = distance_L2

        if self.feature_labels[-1] == "numerical":
            self.prediction_function = regress
        elif self.feature_labels[-1] == "categorical":
            self.prediction_function = classify
        
        self.k = k
        self.predictions = []
        self.priority_queue = []
        self.variant = variant

    def ball_tree_procedure(self, test_dataset):
        """Set up ball tree and run search.

        Args :
            test_dataset (2D list) : Test dataset to use as query
                points.

        Returns :
            predictions (1D list) : List of predictions matching the
                number of test data points.
        """
        
        train_ball_tree = Ball_Tree(self.dataset).root_node
        predictions = []

        for row in test_dataset:
            self.priority_queue = []
            self.ball_tree_search(train_ball_tree, row)
            predictions.append(
                self.prediction_function(self.priority_queue))

        return predictions

    def ball_tree_search(self, ball_tree, point):
        """Ball tree recursive search.

        Steps :
        - Set variables related to distance : distance_query_to_node
          is positive when point is closer to the center of the ball
          tree than the radius of said ball tree. distance_relevant is
          the max distance between point and one of the k nearest
          neighbors found so far.
        - Check if distance_query_to_node is smaller than
          distance_relevant.
        - If so, then if there is either a point or a set of duplicate
          points, then just add them to the "priority queue". If there
          is not just a point or duplicates, then search the left and
          right children, closest first.

        Args :
            ball_tree (node) : Root node of sub ball tree. Not
                necessarily the entire ball tree.
            point (1D list) : Query point.
        """

        if ball_tree.pivot == None:
            distance_query_to_node = (
                self.distance_function(ball_tree.point[:-1], point[:-1])
                - ball_tree.radius
            )
        else:
            distance_query_to_node = (
                self.distance_function(ball_tree.pivot, point[:-1])
                - ball_tree.radius
            )
        
        if len(self.priority_queue) < self.k:
            distance_relevant = distance_query_to_node + 1
        else:
            distances = [self.distance_function(point[:-1], neighbor)
                         for neighbor in self.priority_queue]
            distance_relevant = max(distances)

        if distance_query_to_node < distance_relevant:
            if ball_tree.point != None:
                if (self.distance_function(point[:-1], ball_tree.point)
                    < distance_relevant):
                    self.priority_queue_append(ball_tree.point, point)
            else:
                left, right = ball_tree.left_child, ball_tree.right_child

                if (left.pivot != None and
                    right.pivot != None and
                    self.distance_function(point[:-1], left.pivot)
                    < self.distance_function(point[:-1], right.pivot)):
                    self.ball_tree_search(left, point)
                    self.ball_tree_search(right, point)
                else:
                    self.ball_tree_search(right, point)
                    self.ball_tree_search(left, point)

    def get_hash_table(self, indices, random_vector):
        """Get a hash table which is a list of stacks.

        Args :
            indices (1D list) : List of indices for getting subsets.
            random_vector (1D list) : Random vector for dot products.

        Returns :
            hash_table (2D list) : Hash table for current set of
                indices.
        """

        hash_table = []
        number_of_buckets = ceil(sqrt(self.rows))

        for row in self.dataset:
            hash_value = self.get_hash_value(indices, random_vector,
                                             row)

            if len(hash_table) == 0:
                hash_table.append([hash_value, row[-1]])
            else:
                hash_value_found = False
                    
                for stack in hash_table:
                    if (not hash_value_found
                        and stack[0] == hash_value):
                        stack.append(row[-1])
                        hash_value_found = True

                if not hash_value_found:
                    hash_table.append([hash_value, row[-1]])

        return hash_table

    def get_hash_value(self, indices, random_vector, row):
        """Get hash value by taking random subset and dot product.

        Args :
            indices (1D list) : List of indices for random subset.
            random_vector (1D list) : Random vector for taking dot
                product.
            row (1D list) : Row of dataset to be hashed.

        Returns :
            hash_value (int or float) : Hash value.
        """
        
        number_of_buckets = ceil(sqrt(self.rows))
        subset = [row[i] for i in indices]
        hash_value = (dot_product(random_vector, subset)
                      % number_of_buckets)

        return hash_value

    def knn(self, test_dataset):
        """KNN procedure using sorting and brute force.
         
        Steps :
        - For every test data point, get the distances.
        - Sort the distances and apply the same transformation to the
          training dataset.
        - Do either classification by majority or regression on the k
          nearest neighbors.

        Taken from "Understanding Machine Learning" Ch. 19 : Nearest
        Neighbor.
         
        Args :
            test_dataset (2D list) : Same format as training dataset.
         
        Returns :
            predictions (1D list) : Predictions corresponding to the
                test data points.
        """
        
        predictions = []
        test_rows = len(test_dataset)

        for test_row in range(test_rows):
            distances = []
            test_point = test_dataset[test_row][:-1]
            train_dataset = self.dataset

            for row in range(self.rows):
                train_point = train_dataset[row][:-1]
                distances.append(
                    self.distance_function(test_point, train_point))

            insertion_sort_parallel(train_dataset, distances)

            predictions.append(
                self.prediction_function(train_dataset[:self.k]))
            
        return predictions

    def lsh(self, test_dataset):
        """Locality Sensitive Hashing procedure.

        Steps :
        - Preprocess the training data.
        - For every query point in test_dataset, get the hash value
          which leads us to a hash bucket.
        - Take the labels of the points in the bucket and add them to
          list_of_close_labels.
        - Use self.prediction_function on list_of_close_labels to get
          the prediction.

        Taken from "Similarity Search in High Dimensions via Hashing".

        Args :
            test_dataset (2D list) : Test dataset.

        Returns :
            predictions (1D list) : Predictions for the test data
                points.
        """

        predictions = []
        hash_tables, indices, random_vector = self.lsh_preprocess()
        number_of_buckets = ceil(sqrt(self.rows))
        number_of_subsets = ceil(sqrt(self.cols))

        for point in test_dataset:
            list_of_close_labels = []
            for i in range(number_of_subsets):
                hash_value = self.get_hash_value(indices[i],
                                                 random_vector,
                                                 point)

                for stack in hash_tables[i]:
                    if stack[0] == hash_value:
                        for j in range(1, len(stack)):
                            if len(list_of_close_labels) < self.k:
                                list_of_close_labels.append(
                                    [0, stack[j]])
            """
            if len(list_of_close_labels) == 0:
                if self.prediction_function == regress:
                    predictions.append(5.5)
            elif len(list_of_close_labels) == 1:
                if self.prediction_function == regress:
                    predictions.append(list_of_close_labels)
            else:
                predictions.append(
                    self.prediction_function(list_of_close_labels))
            """
            
            predictions.append(
                self.prediction_function(list_of_close_labels))

        return predictions

    def lsh_preprocess(self):
        """Preprocess the training dataset.

        Take random subsets and hash data points.

        Returns :
            hash_tables (3D list) : List of 2D lists which are hash
                tables.
            list_of_lists_of_indices (2D list) : List of lists of
                indices we will need later.
            random_vector (1D list) : We will need this later so we
                are returning it.
        """
        
        hash_tables = []
        list_of_lists_of_indices = []
        number_of_buckets = ceil(sqrt(self.rows))
        number_of_subsets = size_of_subsets = ceil(sqrt(self.cols))
        random_vector = choices(range(number_of_buckets),
                                k=size_of_subsets)

        for i in range(number_of_subsets):
            indices = sample(range(self.cols - 1), k=size_of_subsets)
            list_of_lists_of_indices.append(indices)
            hash_tables.append(self.get_hash_table(indices,
                                                   random_vector))

        return hash_tables, list_of_lists_of_indices, random_vector

    def predict(self, test_dataset):
        """Get predictions and compare them to the "ground truth".
         
        Args :
            test_dataset (2D list) : Same format as the training
                dataset.
        """
        
        predictions = self.predictor_dispatch(test_dataset)
        test_outputs = [
            test_dataset[x][-1]
            for x in range(len(test_dataset))
        ]

        if self.prediction_function == regress:
            self.print_results(
                mean_squared_error(predictions, test_outputs))
        elif self.prediction_function == classify:
            self.print_results(
                classify_accuracy(predictions, test_outputs))

    def predictor_dispatch(self, test_dataset):
        """Dispatch function for variants of KNN.
         
        Args :
            test_dataset (2D list) : Y'all already know.
         
        Returns :
            predictions (1D list) : Predictions corresponding to the
                test data points.
        """

        if self.variant == "ball_tree":
            predictions = self.ball_tree_procedure(test_dataset)
        elif self.variant == "knn":
            predictions = self.knn(test_dataset)
        elif self.variant == "lsh":
            predictions = self.lsh(test_dataset)

        return predictions

    def print_results(self, value):
        """Print formatted string of results.
         
        Args :
            value (float) : Either mean squared error or accuracy.
         
        Returns :
            (string) : Print results of KNN.
        """
        
        if self.prediction_function == regress:
            print(f"{"Mean Squared Error : ":>22}{value:.3f}")
        elif self.prediction_function == classify:
            print(f"{"Accuracy : ":>22}{value:.3f}")

    def priority_queue_append(self, append_point, query_point):
        """Append point to the "priority queue".

        Remove the furthest point from the query point if there are
        more than k elements.

        Args :
            append_point (1D list) : Point to be appended. Guaranteed
                not to be the one removed.
            query_point (1D list) : Point with which all other points
                are compared to measure distance.
        """
        
        self.priority_queue.append(append_point)

        if len(self.priority_queue) > self.k:
            distances = [self.distance_function(query_point, neighbor)
                         for neighbor in self.priority_queue]
            self.priority_queue.pop(distances.index(max(distances)))

def test_knn():
    """Test function that makes a dataset and runs KNN.

    Sets all the necessary inputs like distance_function and k and
    also measure the time the method takes. May try out real world
    datasets in the future.
    """

    distance_function = "L1"
    k = 8
    dataset = [["numerical", "numerical"]]
    variant = "ball_tree"

    n = 2048
    i = floor(n * 0.9)

    for _ in range(n):
        x = random() * 10
        dataset.append([x, pow(x, 2.0) + (random() * 2.0 - 1.0)])

    dataset = bin_and_one_hot_encode(dataset, 32)
    
    train_dataset = dataset[:i]
    test_dataset = dataset[i:]

    start_time = perf_counter()
    print(variant)
    predictor = Nearest_Neighbors(train_dataset, distance_function, k,
                                  variant)
    predictor.predict(test_dataset)
    end_time = perf_counter()
    
    print(f"{"Time : ":>22}{end_time - start_time:.3f} seconds")
    
    variant = "knn"

    start_time = perf_counter()
    print(variant)
    predictor = Nearest_Neighbors(train_dataset, distance_function, k,
                                  variant)
    predictor.predict(test_dataset)
    end_time = perf_counter()
    
    print(f"{"Time : ":>22}{end_time - start_time:.3f} seconds")
    
    variant = "lsh"

    start_time = perf_counter()
    print(variant)
    predictor = Nearest_Neighbors(train_dataset, distance_function, k,
                                  variant)
    predictor.predict(test_dataset)
    end_time = perf_counter()
    
    print(f"{"Time : ":>22}{end_time - start_time:.3f} seconds")
