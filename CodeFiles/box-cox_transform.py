#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 08:12:53 2019

@author: adityabhat
"""

import pandas as pd

from scipy.stats import boxcox
from matplotlib import pyplot
dataframe = pd.read_csv('/Users/adityabhat/Downloads/Datasets/ScoresRelatedDatasets/Scores_Transformed.csv')
#dataframe.columns = ['diff_']
dataframe['LR'], lam = boxcox(dataframe['LR'])
print('Lambda: %f' % lam)
pyplot.figure(1)
# line plot
pyplot.subplot(211)
pyplot.plot(dataframe['diff_scores'])
# histogram
pyplot.subplot(212)
pyplot.hist(dataframe['diff_scores'])
pyplot.show()
dataframe.to_csv("scores_Transformedv1.csv")