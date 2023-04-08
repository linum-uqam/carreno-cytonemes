# -*- coding: utf-8 -*-

def balanced_class_weights(instances):
    """
    Give balanced weights for labels following the same formula as
    `sklearn.utils.class_weight.compute_class_weight`:
    (total_instances / nb_classes) * (1 / class_instances)
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


if __name__ == "__main__":
    import unittest

    class TestWeights(unittest.TestCase):
        def test_balanced_class_weights(self):
            instances = [8, 4, 2]
            avg = 14 / 3
            expected = [avg * i for i in (0.125, 0.25, 0.5)]
            self.assertTrue(expected, balanced_class_weights(instances=instances))
            
    unittest.main()