# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 17:14:23 2019

@author: smorandv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def rm_ext_and_nan(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A dictionary of clean CTG called c_ctg
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------

    c_ctg = {}
    i = ['LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DR', 'DP', 'ASTV', 'MSTV', 'ALTV', 'MLTV',
         'Width', 'Min', 'Max', 'Nmax', 'Nzeros', 'Mode', 'Mean', 'Median', 'Variance', 'Tendency']
    for x in i:
        q = pd.to_numeric (CTG_features[x], errors='coerce')
        c_ctg[x] = q[~np.isnan (q)]

    del c_ctg[extra_feature]
    # --------------------------------------------------------------------------
    return c_ctg


def nan2num_samp(CTG_features, extra_feature):
    """
    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A pandas dataframe of the dictionary c_cdf containing the "clean" features
    """
    c_cdf = {}
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------

    i = ['LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DR', 'DP', 'ASTV', 'MSTV', 'ALTV', 'MLTV',
         'Width', 'Min', 'Max', 'Nmax', 'Nzeros', 'Mode', 'Mean', 'Median', 'Variance', 'Tendency']
    for x in i:
        Q = CTG_features[x]
        Q = pd.to_numeric (Q, errors='coerce')
        idx_na = Q.index[Q.isna ()].tolist ()
        for ii in idx_na:
            Q.iloc[ii-1] = np.random.choice(Q[~np.isnan (Q)])
        c_cdf[x] = Q

    del c_cdf[extra_feature]
    # -------------------------------------------------------------------------
    return pd.DataFrame(c_cdf)



def sum_stat(c_feat):
    """

    :param c_feat: Output of nan2num_cdf
    :return: Summary statistics as a dicionary of dictionaries (called d_summary) as explained in the notebook
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    stat = c_feat.describe()

    stat = stat.rename({'25%': 'Q1', '50%': 'median', '75%': 'Q3'}, axis='index')
    five_stat = stat.drop(['count', 'mean', 'std'])
    d_summary = five_stat.to_dict()
    # -------------------------------------------------------------------------
    return d_summary


def rm_outlier(c_feat, d_summary):
    """

    :param c_feat: Output of nan2num_cdf
    :param d_summary: Output of sum_stat
    :return: Dataframe of the dictionary c_no_outlier containing the feature with the outliers removed
    """
    c_no_outlier = {}
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------

    i = ['LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'ASTV', 'MSTV', 'ALTV', 'MLTV',
         'Width', 'Min', 'Max', 'Nmax', 'Nzeros', 'Mode', 'Mean', 'Median', 'Variance', 'Tendency']
    for x in i:
        IQR = d_summary[x]['Q3'] - d_summary[x]['Q1']
        TH1 = d_summary[x]['Q1'] - 1.5 * IQR
        TH2 = d_summary[x]['Q3'] + 1.5 * IQR
        c_no_outlier[x]=c_feat[x][(c_feat[x] >= TH1) & (c_feat[x] <= TH2) ]
    # -------------------------------------------------------------------------
    return pd.DataFrame(c_no_outlier)


def phys_prior(c_cdf, feature, thresh):
    """

    :param c_cdf: Output of nan2num_cdf
    :param feature: A string of your selected feature
    :param thresh: A numeric value of threshold
    :return: An array of the "filtered" feature called filt_feature
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:-----------------------------
    filt_feature = c_cdf[feature][(c_cdf[feature] < thresh)]
    # -------------------------------------------------------------------------
    return filt_feature


def norm_standard(CTG_features, selected_feat=('LB', 'ASTV'), mode='none', flag=False):
    """

    :param CTG_features: Pandas series of CTG features
    :param selected_feat: A two elements tuple of strings of the features for comparison
    :param mode: A string determining the mode according to the notebook
    :param flag: A boolean determining whether or not plot a histogram
    :return: Dataframe of the normalized/standardazied features called nsd_res
    """
    x, y = selected_feat
    nsd_res = {}
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    feat1_data = CTG_features[x]
    feat2_data = CTG_features[y]

    x1_mean = np.mean(feat1_data)
    x1_min = min(feat1_data)
    x1_max = max(feat1_data)
    x1_sd = np.std(feat1_data)

    x2_mean = np.mean(feat2_data)
    x2_min = min(feat2_data)
    x2_max = max(feat2_data)
    x2_sd = np.std(feat2_data)

    if mode == 'standard':

        nsd_res[x] = (feat1_data-x1_mean)/x1_sd

        nsd_res[y] = (feat2_data-x2_mean)/x2_sd

    elif mode == 'MinMax':

        nsd_res[x] = (feat1_data-x1_min)/(x1_max-x1_min)

        nsd_res[y] = (feat2_data - x2_min) / (x2_max - x2_min)


    elif mode == 'mean':

        nsd_res[x] = (feat1_data - x1_mean) / (x1_max - x1_min)

        nsd_res[y] = (feat2_data - x2_mean) / (x2_max - x2_min)

    else:
        nsd_res[x] = feat1_data
        nsd_res[y] = feat2_data

    if flag:

        axarr = pd.DataFrame(nsd_res).hist(bins=100, layout=(1, 2), figsize=(20, 10))
        for i, ax in enumerate(axarr.flatten()):
            ax.set_xlabel([selected_feat[i], mode])
            ax.set_ylabel("Count")

        plt.show()


    # -------------------------------------------------------------------------
    return pd.DataFrame(nsd_res)
