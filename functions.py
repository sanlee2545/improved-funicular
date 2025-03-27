"""Functions such as distance or insertion sort."""

from math import pow, sqrt
from random import randint, random
from statistics import quantiles

def bin_and_one_hot_encode(dataset, n):
    """Put numerical features into n bins and one hot encode.

    Args :
        dataset (2D list) : Full dataset including feature labels.
        n (int) : Number of bins.

    Returns :
        new_dataset (2D list) : Dataset with feature labels and bins.
    """

    feature_labels = []
    feature_matrix = []
    new_dataset = []

    for i in range(len(dataset[0][:-1])):
        current_column = [dataset[j][i]
                          for j in range(1, len(dataset))]
        current_feature_label = dataset[0][i]

        if current_feature_label == "categorical":
            vocabulary = get_vocabulary(current_column)

            for k in range(len(vocabulary)):
                feature_labels.append("categorical")
                new_column = []
                
                for value in current_column:
                    if value == vocabulary[k]:
                        new_column.append(1)
                    else:
                        new_column.append(0)

                feature_matrix.append(new_column)
        else:
            bins = quantiles(current_column, n=n)

            for k in bins:
                feature_labels.append("categorical")
                new_column = []

                for value in current_column:
                    if value < k:
                        new_column.append(1)
                    else:
                        new_column.append(0)

                feature_matrix.append(new_column)

            feature_labels.append("categorical")
            new_column = []
            
            for value in current_column:
                if value >= max(bins):
                    new_column.append(1)
                else:
                    new_column.append(0)

            feature_matrix.append(new_column)

    current_column = []

    for i in range(1, len(dataset)):
        current_column.append(dataset[i][-1])

    feature_matrix.append(current_column)

    feature_matrix = transpose(feature_matrix)
    feature_labels.append(dataset[0][-1])
    new_dataset.append(feature_labels)

    return new_dataset + feature_matrix

def classify(data_points):
    """Find majority label.

    Args :
        data_points (1D list) : Data points corresponding to k nearest
            neighbors.

    Returns :
        majority_output (int or string) : Can be either int or string
            corresponding to categorical data. Think of {"a", "b", "c"}
            or {1, 2, 3}.
    """
    
    output_dictionary = []
    outputs = []

    for data_point in data_points:
        outputs.append(data_point[-1])
        if data_point[-1] not in output_dictionary:
            output_dictionary.append(data_point[-1])

    highest_count = outputs.count(output_dictionary[0])
    majority_output = output_dictionary[0]
        
    for output in output_dictionary[1:]:
        if outputs.count(output) > highest_count:
            highest_count = outputs.count(output)
            majority_output = output

    return majority_output

def classify_accuracy(point1, point2):
    """Get accuracy based on correct predictions.

    Args :
        point1 (1D list) : A row of the dataset.
        point2 (1D list) : Another row of the dataset.

    Returns :
        (float) : Accuracy from 0.0 to 1.0 .
    """
    
    correct_predictions = 0
    total_predictions = len(point1)

    for col in range(total_predictions):
        if point1[col] == point2[col]:
            correct_predictions += 1

    return float(correct_predictions) / total_predictions

def distance_L1(point1, point2):
    """L1 norm.

    Args :
        point1 (1D list) : A row of the dataset.
        point2 (1D list) : Another row of the dataset.

    Returns :
        distance (int or float) : Depending on the data types of the
            inputs.
    """
    
    distance = 0

    for col in range(len(point1)):
        distance += abs(point1[col] - point2[col])

    return distance

def distance_L2(point1, point2):
    """L2 norm.

    Args :
        point1 (1D list) : A row of the dataset.
        point2 (1D list) : Another row of the dataset.

    Returns :
        (float or int) : Depending on the data types of the inputs.
    """
    
    distance = 0

    for col in range(len(point1)):
        distance += pow(point1[col] - point2[col], 2.0)

    return sqrt(distance)

def dot_product(list1, list2):
    """Multiply two lists element wise like a dot product.

    Args :
        list1 (1D list) : First list.
        list2 (1D list) : Second list.

    Returns :
        result (int or float) : Dot product.
    """

    result = 0

    for i in range(len(list1)):
        result += list1[i] * list2[i]

    return result

def get_vocabulary(data_points):
    """Return list of distinct values in data_points.

    Args :
        data_points (1D list) : List of data points.

    Returns :
        vocabulary (1D list) : List of distinct values in data_points.
    """

    vocabulary = []

    for point in data_points:
        if point not in vocabulary:
            vocabulary.append(point)

    return vocabulary

def insertion_sort_parallel(dataset, distances):
    """Insertion sort on distances and dataset in parallel.

    Taken from "Introduction to Algorithms" by CLRS.

    Args :
        dataset (2D list) : Training dataset rows including labels.
        distances (1D list) : Distances from a particular point to each
            point in the training dataset.
    """
    
    for j in range(1, len(distances)):
        key_dataset = dataset[j]
        key_distance = distances[j]
        i = j - 1

        while i >= 0 and distances[i] > key_distance:
            dataset[i + 1] = dataset[i]
            distances[i + 1] = distances[i]
            i -= 1

        dataset[i + 1] = key_dataset
        distances[i + 1] = key_distance
            
def mean_squared_error(point1, point2):
    """Mean squared error.

    Args :
        point1 (1D list) : A row in the dataset.
        point2 (1D list) : Another row in the dataset.

    Returns :
        mse (float) : Mean squared error.
    """
    
    mse = 0
    total_predictions = len(point1)

    for col in range(total_predictions):
        mse += pow(point1[col] - point2[col], 2.0)

    mse /= float(total_predictions)

    return mse
    
def regress(data_points):
    """Average of label values.

    Args :
        data_points (2D list) : List of k data points.

    Returns :
        mean (float) : Mean which is used as a prediction.
    """
    
    mean = 0

    for data_point in data_points:
        mean += data_point[-1]

    mean /= float(len(data_points))

    return mean

def transpose(matrix):
    """Transpose a matrix.

    Taken from Python Tutorial Section 5.1.4. : Nested List
    Comprehensions.

    Args :
        matrix (2D list) : Matrix to be transposed.

    Returns :
        (2D list) : Transposed matrix.
    """

    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]
