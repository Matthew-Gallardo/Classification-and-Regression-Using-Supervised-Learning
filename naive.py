import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation

from utility import visualize_classifier

#File containing data
input_file ='data_multivar_nb.txt'
