"""
* Ball Tree and Node classes.
"""

from functions import distance_L2
from random import randint, random
from statistics import median, variance

class Ball_Tree:
    """Ball tree data structure.

    Taken from wikipedia.org/wiki/Ball_tree .

    Attributes :
        root_node (node) : Pointer to root node.
    """
    
    def __init__(self, dataset):
        """Initialize ball tree.

        Args :
            dataset (2D list) : Training dataset.
        """
        
        self.root_node = self.make_ball_tree(dataset)

    def find_greatest_spread(self, dataset):
        """Return column in dataset with highest variance.

        Args :
            dataset (2D list) : Data set which may be a subset of the
                full training dataset.

        Returns :
            col_with_highest_variance (int) : Index of column.
        """
        
        col_with_highest_variance = 0
        highest_variance = 0
        
        for col in range(len(dataset[0])):
            current_variance = variance([row[col] for row in dataset])

            if current_variance > highest_variance:
                highest_variance = current_variance
                col_with_highest_variance = col

        return col_with_highest_variance

    def get_children(self, column, dataset, med):
        """Children refers to the left and right child nodes.

        Args :
            column (int) : Index of column which is being focused.
            dataset (2D list) : Dataset.
            med (float or int) : Center value of column.

        Returns :
            left (2D list) : Set of points to put into left child.
            right (2D list) : Set of points to put into right child.
        """
        
        left = right = None

        for row in dataset:
            if row[column] <= med:
                if left == None:
                    left = [row]
                else:
                    left.append(row)
            else:
                if right == None:
                    right = [row]
                else:
                    right.append(row)

        if left == None or right == None:
            left = right = None

            for row in dataset:
                if row[column] < med:
                    if left == None:
                        left = [row]
                    else:
                        left.append(row)
                else:
                    if right == None:
                        right = [row]
                    else:
                        right.append(row)

        return left, right

    def get_radius(self, dataset, pivot):
        """Get max distance between points in dataset and pivot.

        Args :
            dataset (2D list) : Set of data points.
            pivot (1D list) : Center of node.

        Returns :
            max_distance (float or int) : Max distance between points
                in dataset and pivot.
        """
        
        max_distance = 0

        for row in dataset:
            current_distance = distance_L2(pivot, row)
            if current_distance > max_distance:
                max_distance = current_distance
                
        return max_distance

    def make_ball_tree(self, dataset):
        """Make ball tree from dataset using recursion.

        Args :
            dataset (2D list) : Dataset.

        Returns :
            node (node) : Root node of current subtree.
        """

        node = Node()
        dataset_no_labels = [row[:-1] for row in dataset]
        
        if len(dataset) == 1:
            node.point = dataset[0]
            node.radius = self.get_radius(dataset_no_labels,
                                          node.point[:-1])
            return node
        else:
            column = self.find_greatest_spread(dataset_no_labels)
            medians = [
                median([row[col] for row in dataset_no_labels])
                for col in range(len(dataset_no_labels[0]))
            ]
            left, right = self.get_children(column,
                                            dataset,
                                            medians[column])
            
            if left == None:
                node.point = right[0]
                node.pivot = right[0][:-1]
                node.radius = self.get_radius(dataset_no_labels,
                                              node.pivot)
                return node
            elif right == None:
                node.point = left[0]
                node.pivot = left[0][:-1]
                node.radius = self.get_radius(dataset_no_labels,
                                              node.pivot)
                return node
            else:
                node.pivot = medians
                node.radius = self.get_radius(dataset_no_labels,
                                              node.pivot)
            
                node.left_child = self.make_ball_tree(left)
                node.right_child = self.make_ball_tree(right)
                
                return node

    def print_ball_tree(self, node):
        """Print nodes of ball tree.

        Args :
            node (node) : Root node of current subtree.

        Returns :
            (string) : Print point of node.
        """
        
        if node != None:
            self.print_ball_tree(node.left_child)
            if node.point != None:
                print(node.point)
            self.print_ball_tree(node.right_child)

class Node:
    """Node for ball tree.

    Attributes :
        left_child (node) : Set of points less than or equal to
            median.
        pivot (1D list) : Center point of node.
        point (1D list) : Contains a point if node has all duplicates
            or just one point.
        radius (float) : Distance from pivot to outer edge of ball of
            node.
        right_child (node) : Set of points greater than or equal to
            median.
    """
    
    def __init__(self):
        self.left_child = None
        self.pivot = None
        self.point = None
        self.radius = None
        self.right_child = None
