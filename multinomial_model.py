"""
    @nghia nh
    ---
    Multinomial Naive Bayes,
    many-to-one assumption,
    EM algorithm

"""

import logging

import numpy as np
from scipy.sparse import csc_matrix
from sklearn.naive_bayes import MultinomialNB


class MultinomialManyToOneEM(MultinomialNB):
    """
    Multinomial Naive Bayes with many-to-one assumption and EM algorithm
    """
    def __init__(self, epsilon=1e-3):
        """
        Follow the ClassifierMixin template of scikit-learn
        :param epsilon: threshold for convergence condition
        """
        MultinomialNB.__init__(self)

        self.component_sum_up = [0]
        self.component_number = 0

        self.epsilon = epsilon
        self.feature_number = 0
        self.classes_ = None

    def _m_step(self, x, label_pr):
        """
        maximization step: re-estimate new parameters
        :param x: data instances (n_instances x n_features)
        :param label_pr: distribution label on data (n_instances x n_components)
        :return:
        """
        self.class_count_ = np.zeros(self.component_number, dtype=np.float64)
        self.feature_count_ = np.zeros(
            (self.component_number, self.feature_number),
            dtype=np.float64
        )
        self._count(x, label_pr)

        alpha = self._check_alpha()
        self._update_feature_log_prob(alpha)
        self._update_class_log_prior()

    def fit(self, x, y_l, component_numbers=None, data_label_pr=None):
        """
        fit input data into model, EM algorithm
        :param x: input data
        :param y_l: labeled for labeled instances
        :param component_numbers: list of component number for each label
        :param data_label_pr: init distribution of components on labeled data
        :return:
        """
        logging.info('MultinomialManyToOneEM: fit: '
                     'component_numbers {}'.format(component_numbers))

        x = np.concatenate((x[0], x[1]), axis=0)  # merge data [x_l | x_u]

        for count in component_numbers:
            self.component_sum_up.append(count + self.component_sum_up[-1])
        self.component_number = self.component_sum_up[-1]
        self.classes_ = np.arange(self.component_number)

        # extract data info
        labeled_number = len(y_l)
        self.feature_number = len(x.T)

        x = csc_matrix(x)

        self._m_step(x[:labeled_number], data_label_pr)
        log_mle_old, log_mle_new = -1, 0

        # EM algorithm
        loop_count = 0
        while abs(log_mle_new - log_mle_old) > self.epsilon:
            logging.info('Loop: {} -- MLE diff = {}'.format(
                loop_count, log_mle_new - log_mle_old
            ))
            loop_count += 1

            delta = self.predict_proba(x)
            # constraint components in same label sum to one
            for i in range(labeled_number):
                start_po = self.component_sum_up[y_l[i]]
                end_po = self.component_sum_up[y_l[i] + 1]
                delta[i, start_po:end_po] /= np.sum(delta[i, start_po:end_po])
                delta[i, :start_po] = 0
                delta[i, end_po:] = 0

            self._m_step(x, delta)

            # check convergence condition
            log_mle_old = log_mle_new
            log_mle_new = np.exp(self._joint_log_likelihood(x)).sum()

        return self

    def predict(self, x):
        """
        Estimated value of x for each class
        :param x: test data
        :return:
        """
        jll = self._joint_log_likelihood(x)
        class_number = len(self.component_sum_up) - 1
        predicted = np.zeros((len(x), class_number))

        # sum up the predicted pr of all components in a label
        for i in range(class_number):
            start_po = self.component_sum_up[i]
            end_po = self.component_sum_up[i + 1]
            predicted[:, i] = np.sum(jll[:, start_po:end_po], axis=1)

        # predict the label that has max pr on its all components
        return np.argmax(predicted, axis=1)
