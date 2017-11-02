import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from logging import StreamHandler, DEBUG, Formatter, getLogger
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import log_loss, roc_auc_score, roc_curve,  auc
from . import load_data

logger = getLogger(__name__)

DIR = 'result_tmp/'
SAMPLE_SUBMIT_FILE = '../input/sample/submission.csv'

def gini(y, pred):
    fpr, tpr, thr = roc_curve(y, pred, pos_label=1)
    g = 2 * auc(fpr, tpr) - 1
    return g
if __name__ == '__main__':
    log_fmt = Formatter('%(asctime)s %(name)s %(lineo)d [%(levelname)s] [%(funcName)s] %(message)s ')
    handler = StreamHandler()
    handler.setLevel('INFO')
    handler.setFormatter(log_fmt)
    logger.addHandler(handler)

    df = load_data.load_train_data()

    x_train = df.drop('target', axis=1)
    y_train = df['target'].values

    use_cols = x_train.columns.values

    cv = StratifiedKFold(n_split=5, shuffle=True, random_state=0)
    all_params = {'C': [10**i for i in range(-1, 2)],
                  'fit_intercept': [True, False],
                  'penalty': ['l2', 'l1'],
                  'random_state': [0]}

