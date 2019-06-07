import sys
import logging

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit

from utilities import ResultExporter
from utilities import ParamSearch
from data.newsgroups_processing import NewsgroupsData
from data.reuters_processing import ReutersData
from multinomial_model import MultinomialManyToOneEM
from sklearn.naive_bayes import MultinomialNB

# log config
LOG_FILE = 'source.log'

logFormatter = logging.Formatter(
    '%(asctime)s [%(threadName)-12.12s] '
    '[%(levelname)-5.5s]  %(message)s'
)
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.DEBUG)

fileHandler = logging.FileHandler(LOG_FILE)
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)


###############################################################################
# Data dir

# Where all data are located, should only one location
data_folder_dir = 'data/'

# Where each data are located in data_folder
data_dir = {
    '20news': data_folder_dir + NewsgroupsData.default_folder,
    'reuters': data_folder_dir + ReutersData.default_folder
}


###############################################################################
# Experiment setup

class ExperimentSetUp:
    def __init__(self, data_processor, result_filename):
        self.cv_fold_number = 5
        self.data_processor = data_processor.transform()
        self.result_filename = result_filename

    def process_ssl(self, test_size=0.3, unlabeled_size=0.7):

        # cumulative results
        result = {
            'nb': ResultExporter('nb'),
            'em_random': ResultExporter('em_random'),
            'em_tree': ResultExporter('em_tree'),
            'em_kmeans': ResultExporter('em_kmeans'),
        }

        # split labeled and unlabeled set
        sl_sss_data = StratifiedShuffleSplit(
            n_splits=self.cv_fold_number,
            test_size=test_size,
            random_state=0
        )

        param_search = ParamSearch()

        for train_index, test_index in sl_sss_data.split(
                np.zeros(self.data_processor.x_number),
                self.data_processor.y):
            x_train, y_train, x_test, y_test = self.data_processor.extract(train_index, test_index)

            ssl_sss_data = StratifiedShuffleSplit(
                n_splits=self.cv_fold_number,
                test_size=unlabeled_size,
                random_state=0)
            for labeled_index, unlabeled_index in ssl_sss_data.split(x_train, y_train):
                x_l, y_l = x_train[labeled_index], y_train[labeled_index]
                x_u, _ = x_train[unlabeled_index], y_train[unlabeled_index]

                # experiments

                nb = MultinomialNB()
                em_random = MultinomialManyToOneEM()
                em_tree = MultinomialManyToOneEM()
                em_kmeans = MultinomialManyToOneEM()

                nb.fit(x_l, y_l)
                components, label_pr = param_search.component_search(x_l, y_l, x_u, type='random')
                em_random.fit(
                    x=[x_l, x_u],
                    y_l=y_l,
                    component_numbers=components,
                    data_label_pr=label_pr
                )
                components, label_pr = param_search.component_search(x_l, y_l, x_u, type='tree')
                em_tree.fit(
                    x=[x_l, x_u],
                    y_l=y_l,
                    component_numbers=components,
                    data_label_pr=label_pr
                )
                components, label_pr = param_search.component_search(x_l, y_l, x_u, type='kmeans')
                em_kmeans.fit(
                    x=[x_l, x_u],
                    y_l=y_l,
                    component_numbers=components,
                    data_label_pr=label_pr
                )

                # sum score on y_test
                report = classification_report(y_test, nb.predict(x_test), output_dict=True)
                result['nb'].sum_report(report)
                report = classification_report(y_test, em_random.predict(x_test), output_dict=True)
                result['em_random'].sum_report(report)
                report = classification_report(y_test, em_tree.predict(x_test), output_dict=True)
                result['em_tree'].sum_report(report)
                report = classification_report(y_test, em_kmeans.predict(x_test), output_dict=True)
                result['em_kmeans'].sum_report(report)

        # export result
        for test_case in result.values():
            test_case.export(self.result_filename, scale=self.cv_fold_number*self.cv_fold_number)


###############################################################################
# Test cases

class Experiment:
    def __init__(self):
        self.experiments_map = {
            'newsgroups': self.newsgroups_experiment,
            'reuters': self.reuters_experiment
        }

        self.params_map = {
            'newsgroups1': {
                'folder_dir': data_dir['20news'],
                'categories': [
                    'comp.graphics', 'comp.os.ms-windows.misc',
                    'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x',
                    'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey'
                ],
                'positive_labels': [
                    'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey'
                ],
                'feature_score': 'tfidf',
                'feature_number': 600,
                'normalize': 'tfidf',
                'scale': 10000.
            },
            'newsgroups2': {
                'folder_dir': data_dir['20news'],
                'categories': [
                    'comp.graphics', 'comp.os.ms-windows.misc',
                    'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x',
                    'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space'
                ],
                'positive_labels': [
                    'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space'
                ],
                'feature_score': 'tfidf',
                'feature_number': 600,
                'normalize': 'tfidf',
                'scale': 10000.
            },
            'newsgroups3': {
                'folder_dir': data_dir['20news'],
                'categories': [
                    'comp.graphics', 'comp.os.ms-windows.misc',
                    'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x',
                    'talk.politics.guns', 'talk.politics.mideast',
                    'talk.politics.misc', 'talk.religion.misc'
                ],
                'positive_labels': [
                    'talk.politics.guns', 'talk.politics.mideast',
                    'talk.politics.misc', 'talk.religion.misc'
                ],
                'feature_score': 'tfidf',
                'feature_number': 600,
                'normalize': 'tfidf',
                'scale': 10000.
            },
            'reuters': {
                'folder_dir': data_dir['reuters'],
                'positive_labels': 'acq',
                'feature_score': 'tfidf',
                'feature_number': 600,
                'normalize': 'tfidf',
                'scale': 10000.
            }
        }

    def get_experiment(self, exp_name, params_name):
        try:
            return self.experiments_map[exp_name](**self.params_map[params_name])
        except KeyError:
            logging.error('Experiment: Non recognized experiment %s.' % exp_name)
            raise

    @staticmethod
    def newsgroups_experiment(folder_dir, categories=None, positive_labels=None,
                              feature_score=None, feature_number=600,
                              normalize=None, **kwargs_normalize):
        """
        Experiment on 20Newsgroup data, binary classifier

        Class names
        'alt.atheism',
        'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
        'comp.sys.mac.hardware', 'comp.windows.x',
        'misc.forsale',
        'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey',
        'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space',
        'soc.religion.christian',
        'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'

        :param folder_dir: data folder
        :param categories: picked categorizes
        :param positive_labels: categorizes that will be used for positive label
        :param feature_score: word scoring method for reducing feature
        :param feature_number: number of feature used (select from top words with high scrore)
        :param normalize: data normalize method (after feature reducing)
        :return:
        """
        logging.info('Start 20Newsgroup Experiment')

        result_file_name = 'newsgroups_experiment.out'
        data_processor = NewsgroupsData(
            folder_dir=folder_dir, categories=categories, positive_labels=positive_labels,
            feature_score=feature_score, feature_number=feature_number,
            normalize=normalize, **kwargs_normalize)

        exp = ExperimentSetUp(data_processor, result_file_name)

        return exp

    @staticmethod
    def reuters_experiment(folder_dir, positive_labels=None,
                              feature_score=None, feature_number=600,
                              normalize=None, **kwargs_normalize):
        """
        Experiment on Reuters 21578 data, binary classifier.
        Number of instances: 19716

        :param folder_dir: data folder
        :param positive_labels: categorizes that will be used for positive label
        :param feature_score: word scoring method for reducing feature
        :param feature_number: number of feature used (select from top words with high scrore)
        :param normalize: data normalize method (after feature reducing)
        :return:
        """
        logging.info('Start Reuters Experiment')

        result_file_name = 'reuters_experiment.out'
        data_processor = ReutersData(
            folder_dir=folder_dir, positive_labels=positive_labels,
            feature_score=feature_score, feature_number=feature_number,
            normalize=normalize, **kwargs_normalize)

        exp = ExperimentSetUp(data_processor, result_file_name)

        return exp


###############################################################################
def main():
    logging.info('Start main()')
    try:
        exp_setup = Experiment()

        exps = [
            exp_setup.get_experiment('newsgroups', 'newsgroups1'),
            exp_setup.get_experiment('newsgroups', 'newsgroups2'),
            exp_setup.get_experiment('newsgroups', 'newsgroups3')
            exp_setup.get_experiment('reuters', 'reuters')
        ]

        for exp in exps:
            exp.process_ssl(test_size=0.5, unlabeled_size=0.97)

    except BaseException:
        logging.exception('Main exception')
        raise

    return 'Done main()'


if __name__ == '__main__':
    status = main()
    sys.exit(status)
