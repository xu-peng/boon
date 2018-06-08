# -*- coding: utf-8 -*-
import re
import numpy as np
import pandas as pd


# Loading functions:

def best_validation_rows(log_df, valid_col='valid_accuracy', second_criterion='iterations_done'):
    """
    Takes a dataframe created by scripts/logs_to_dataframe.py and returns
    a dataframe containing the best-validation row for each log.
    """
    return log_df.sort_values([valid_col,second_criterion],ascending=False).drop_duplicates(['log'])
    
def load_and_preprocess(df_pkl,valid_col='valid_accuracy',second_criterion='iterations_done'):
    all_epochs=pd.read_pickle(df_pkl)
    best_val_rows=best_validation_rows(all_epochs, valid_col, second_criterion)
    return best_val_rows
    
def parse_hyperparams(df,params=['ehd','sed','b']):
    df.loc[:,'params']=[re.search('\/\d+\_([^\/]*)\.out',lp).group(1) for lp in df['log']]
    for param in params:
        df.loc[:,param]=[int(re.search('-'+param+'_(\d+)',lp).group(1)) for lp in df['params']]
        
        
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Functions for loading specific results tables:

def cbt_fh():
    """
    Load results from experiments with fixed hyperparameters
    """
    cbt=pd.read_csv('results/cbt_fixed_hyppar.csv')
    
    cbt=cbt.rename(columns={'valid.accuracy':'valid_accuracy'})
    
    # Filter out models which crashed before finishing the first epoch:
    cbt=cbt.query('epoch>0')
    
    # Parse the parameter string and experiment name from the log path:
    cbt.loc[:,'params']=[re.search('\/\d+\_([^\/]*)\.out',lp).group(1) for lp in cbt['log_path']]
    cbt.loc[:,'experiment']=[re.search('\/cbt-([^\/]*)\/[^\/]*\d+\/\d+\_([^\/]*)\.out',lp).group(1) for lp in cbt['log_path']]
    
    bv=cbt.query('experiment=="big-valid"')
    cbd=cbt.query('experiment=="dist"')
    
    cbt = cbd
    
    return cbt, bv


def cbt_rs():
    cbtrs = load_and_preprocess('results/cbtrs_results.pkl')
    parse_hyperparams(cbtrs)
    # Filter out trainings that crashed straight away:
    cbtrs = cbtrs.query("iterations_done > 0")
    return cbtrs

def cifar10_densenet():
    return pd.read_pickle('results/cifar10_densenet_depth40_results.pkl')

def cifar100_resnet(min_steps=35000):
    resnet_results = pd.read_pickle('results/cifar100-resnet-oct17.pkl')
    # Filter out unfinished 
    resnet_results = resnet_results.query('index>={}'.format(min_steps))
    return resnet_results
  