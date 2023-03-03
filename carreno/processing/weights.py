# -*- coding: utf-8 -*-
import numpy as np

def balanced_class_weights(instances):
    """
    Give balanced weights for labels following this formula :
    (1 / class_instances) * (total_instances / nb_classes)
    unless class_instances is 0, then the weight is 0 and it isn't included in nb_classes
    Parameters
    ----------
    instances : [class instances]
        list of the number of class instances
    Returns
    -------
    weights : [float]
        weight for each classes
    """
    n_class = len(instances) - sum([i == 0 for i in instances])
    total = sum(instances)
    fair_distribution = total / n_class
    weights = [0 if i == 0 else fair_distribution / i for i in instances]
                
    return weights
