import logging

import numpy as np
from scipy.sparse import csc_matrix
from sklearn.naive_bayes import MultinomialNB


class MultinomialManyToOneEM(MultinomialNB):
    def __init__(self, epsilon=1e-3):
        MultinomialNB.__init__(self)

        self.component_sum_up = [0]
        self.component_number = 0

        self.epsilon = epsilon
        self.feature_number = 0
        self.classes_ = None

    def _m_step(self, x, label_pr):
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
        """Estimated value of x for each class"""
        jll = self._joint_log_likelihood(x)
        class_number = len(self.component_sum_up) - 1
        predicted = np.zeros((len(x), class_number))

        for i in range(class_number):
            start_po = self.component_sum_up[i]
            end_po = self.component_sum_up[i + 1]
            predicted[:, i] = np.sum(jll[:, start_po:end_po], axis=1)

        return np.argmax(jll, axis=1)
