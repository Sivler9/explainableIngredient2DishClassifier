"""
TODO - docs
"""
import numpy as np


def kg_matrix(kg_dict, features=None, classes=None):
    """
    Create knowledge graph (adjacency matrix) from class map (dict)

    :param dict kg_dict: Class-to-Feature map
    :param list features: Pre-computed features list (micro/parts)
    :param list classes: Pre-computed class list (macro/object)
    :returns np.ndarray: Adjacency Matrix
    """
    if classes is None:
        classes = sorted(kg_dict.keys())
    if features is None:
        features = sorted(set([part for clas in kg_dict.values() for part in clas]))
    adjacency_matrix = -np.ones((len(features), len(classes)), dtype=np.int8)
    for i, part in enumerate(features):
        for k, clas in enumerate(classes):
            if part in kg_dict[clas]:
                adjacency_matrix[i, k] = 1
    return adjacency_matrix


def kg_build(dictionary, classes, parts):
    """
    TODO - docs

    :param dictionary:
    :param classes:
    :param parts:
    :return:
    """
    graph = {}
    for clas, part in dictionary.items():
        clas, part = f'c{classes.index(clas)}', set([f'p{parts.index(n)}' for n in part])
        graph[clas] = part
        for p in part:
            if p not in graph:
                graph[p] = {clas}
            else:
                graph[p].add(clas)
    return graph


def tests():  # TODO - move tests to their own directory
    raise NotImplementedError


if __name__ == '__main__':
    tests()
