Multinomial Naive Bayes with EM algorithm
===

This project implements the multinomial Naive Bayes with 
[Nigram's many-to-one assumption][1] EM algorithm (semi-supervised learning). 

The implement of EM algorithm derives from the basic multinomial Naive Bayes 
of *scikit-learn*'s `MultinomialNB`.


How to reproduce
---
- Install requirements.txt
- All test cases are implemented in `source.py` as individual functions.

Main modules
---
- `data/...` all data processing scripts
- `multinomial_model` implements model, 
- `utilities` supports some simple graph construction methods

[1]: http://www.kamalnigam.com/papers/emcat-mlj99.pdf