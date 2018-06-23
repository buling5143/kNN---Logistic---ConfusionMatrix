# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 10:10:32 2018

@author: Utpal
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 11:38:14 2018

"""
"""

Created on %(date)s
@author: %(username)s

1. Create a heat map/co-orelation matrix
2. Regression (report: R-square) - use Logit to predict TP/FP or regular to predict  score
3. In the regression look for t-stat to see if the co-offficinet is stat significant
4. In the regression can you use dummy variables for control - figure out which ones
5. Generate a distribution for the predictors 

======

6. Use PCA/Factor to see which predictors really matter
7. Re-run regression?

"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import numpy as np
#import scipy.stats as stat
#import matplotlib.rcsetup as rcsetup
#from sklearn.preprocessing import scale
#from sklearn.decomposition import PCA
#from IPython.display import display, HTML

#from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

import statsmodels.api as sm

DISPLAY_MAX_ROWS = 50  # number of max rows to print for a DataFrame
pd.set_option('display.max_rows', DISPLAY_MAX_ROWS)

#from StatLib import *

#==============================================================================
# Colorization of sys.stderr (standard Python interpreter)
#==============================================================================

def runLogit():
      
        df = pd.read_excel('abc.xlsx', sheetname='InputToCode')
       
        evaluation_code= df['evaluation_code']
        source_video     = df['video']
        source_review    = df['review']
        source_instagram = df['instagram']
#       source_news      = df['news']
#       source_forum     = df['forum']
        source_facebook  = df['facebook']
        source_general   = df['general']
        sentiment = df['document_sentiment']
#       EntTop_distance = df['EntTop_distance']
#       RiskScore_non_stat = df['RiskScore_non_stat']
#       words_per_sentence  = df['WordsPer_Sentence']
#       sentence_count  	= df['sentence_count']
#       word_count = df['word_count']
                       
        df = pd.DataFrame(
                          {
                           'evaluation_code': evaluation_code,       
                           'source_video': source_video,  	
		                     'source_review': source_review,
                           'source_instagram': source_instagram,
#                          'source_news': source_news,
#                          'source_forum': source_forum,  	
                           'source_facebook': source_facebook, 	
                           'source_general': source_general,   
                           'sentiment': sentiment,
#                           'EntTop_distance': EntTop_distance,
#                           'RiskScore_non_stat': RiskScore_non_stat,	
#                           'words_per_sentence': words_per_sentence, 	
#                           'sentence_count': sentence_count,	
#                           'word_count':	word_count
                           }                           
                        )
       
        Correlation_Matrix = df.corr()
        print ("\n ***** Co-orelation matrix*****::\n\n", Correlation_Matrix)  
       
        plt.style.use('classic')
        plt.figure()
        plt.show()
        sns.heatmap(Correlation_Matrix, vmax=1., square=True)
       
        #data_final = df.columns.values.tolist()       
        #print (data_final)       
       
        y = df['eval_relevance_code'].values
        #y = np.arange(1, 611)
        print (len(y))
        print (df.shape)
             
       #To select the best predictor variables 
       #Feature selection
       
        logistic = LogisticRegression()       
        rfe = RFE(logistic, 7)
        rfe = rfe.fit(y.astype(float), df.astype(float))
        #rfe = rfe.fit(y, df)
        print(rfe.support_)
        print(rfe.ranking_)
       
       
       
# Reference: 
#     
# (1) https://stackoverflow.com/questions/33833832/building-multi-regression-model-throws-error-pandas-data-cast-to-numpy-dtype-o
# (2) https://stackoverflow.com/questions/21234539/statsmodels-ols-function-for-multiple-regression-parameters
#     
#        logit_model = sm.Logit(y.astype(float), df.astype(float))
#        result = logit_model.fit()
#        print (result.summary())
       
       
       

#==============================================================================
# Initial call
#==============================================================================

runLogit()