import snapatac2 as sa2
import numpy as np
import scipy
import pandas as pd
from sklearn import metrics
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics.cluster import pair_confusion_matrix

# test
true_label = [0,0,1,1,2,3]
pred_label = [10,10,2,2,3,0]
metrics.rand_score(true_label, pred_label)
metrics.adjusted_rand_score(true_label, pred_label)
contingency_matrix(true_label, pred_label)
pair_confusion_matrix(true_label, pred_label)
