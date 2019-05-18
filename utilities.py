import logging
import copy

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score

from multinomial_model import MultinomialManyToOneEM


class ResultExporter:
    __REPORT_FORM = {
        '0': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
        '1': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
        'macro avg': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0}
    }

    def __init__(self, name):
        self.report = ResultExporter.report_form()
        self.name = name

    @staticmethod
    def report_form():
        return copy.deepcopy(ResultExporter.__REPORT_FORM)

    def sum_report(self, x):
        """
        sum in self.report a classification_report
        :param x: classification_report
        :return:
        """
        x.pop('weighted avg', None)
        x.pop('micro avg', None)
        x.pop('accuracy', None)

        for n1, n2 in zip(self.report, x):
            for v1, v2 in zip(self.report[n1], x[n2]):
                self.report[n1][v1] = self.report[n1][v1] + x[n2][v2]

    def export(self, filename, scale=1.0, message=None):
        """
         Export self.report to file
        :param filename: file to export
        :param scale: scale of export result
        :param message: name of export content
        :return:
        """
        # TODO verify datatype of input parameters
        with open(filename, 'a') as f:
            if message is not None:
                f.writelines('\n' + message)
            f.writelines('\n' + self.name + '\n')
            f.write('\n{:>15}{:>10}{:>10}{:>10}{:>10}\n'.format(
                '', 'precision', 'recall', 'f1-score', 'support'
            ))
            for att in self.report:
                f.write('{:>15}'.format(att))
                for val in self.report[att]:
                    f.write('{:10.2f}'.format(
                        round(self.report[att][val] / scale, 2)
                    ))
                f.write('\n')


class RandomAssignment:
    def __init__(self):
        self.labels = None
        self.data_number = 0

    @staticmethod
    def _equal_sampling(component_number):
        """
        Return a list of component_number uniformed numbers with constraint sum all element is 1
        :param component_number: number of components
        :return: list of randomly sampling component
        """

        samples = np.random.uniform(0, 1, component_number - 1)
        samples = np.append(samples, [0, 1])
        samples.sort()
        for i in range(len(samples) - 1):
            samples[i] = samples[i + 1] - samples[i]
        return samples[:-1]

    def fit(self, labels, x=None):
        self.labels = labels
        self.data_number = len(labels)
        return self

    def get_assignment(self, component_numbers):
        component_sum_up = [0]
        for count in component_numbers:
            component_sum_up.append(count + component_sum_up[-1])

        assignment = np.zeros((self.data_number, component_sum_up[-1]))

        for i in range(self.data_number):
            start_po = component_sum_up[self.labels[i]]
            end_po = component_sum_up[self.labels[i]+1]
            assignment[i, start_po:end_po] = self._equal_sampling(end_po - start_po)

        return assignment


class TreeAssignment:
    def __init__(self):
        self.x = None
        self.labels =None
        self.data_number = 0
        self.trees = []

    def fit(self, labels, x):
        self.x = x
        self.labels = labels
        self.data_number = len(x)
        return self

    def get_assignment(self, component_numbers):
        component_sum_up = [0]
        for count in component_numbers:
            component_sum_up.append(count + component_sum_up[-1])

        assignment = np.zeros((self.data_number, component_sum_up[-1]))

        # FIXME Bad kitty! extract tree 1 time so that can get
        #  different component_numbers setups later
        for label_id in range(len(component_numbers)):
            label_indices = (self.labels == label_id)

            tree = AgglomerativeClustering(
                linkage='average',
                n_clusters=component_numbers[label_id]
            ).fit(self.x[label_indices])

            start_po = component_sum_up[label_id]
            assignment[label_indices, tree.labels_ + start_po] = 1

        return assignment


class ParamSearch:
    def __init__(self):
        self.MAX_COMPONENT = 10
        self.search_map = {
            'random': RandomAssignment,
            'tree': TreeAssignment
        }

    def component_search(self, x_l, y_l, x_u, type='random'):
        logging.info('ParamSearch: {} assignment'.format(type))

        assignment = self.search_map[type]()
        max_score = 0
        max_components = None

        for i in range(1, self.MAX_COMPONENT):
            for j in range(1, self.MAX_COMPONENT):
                sss_data = StratifiedShuffleSplit(
                    n_splits=1,
                    test_size=0.5,
                    random_state=0
                )
                train_index, test_index = sss_data.split(x_l, y_l).__next__()

                components = [i, j]

                assignment.fit(y_l[train_index], x_l[train_index])
                label_pr = assignment.get_assignment(components)
                y_predict = MultinomialManyToOneEM().fit(
                    x=[x_l[train_index], x_u],
                    y_l=y_l[train_index],
                    component_numbers=components,
                    data_label_pr=label_pr
                ).predict(x_l[test_index])

                score = accuracy_score(y_l[test_index], y_predict)
                if score > max_score:
                    max_score = score
                    max_components = components

        logging.info('Selected Component: {}'.format(max_components))
        return max_components, assignment.fit(y_l, x_l).get_assignment(max_components)
