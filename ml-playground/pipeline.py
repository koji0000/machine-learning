import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import log_loss, roc_auc_score, roc_curve, auc
import xgboost as xgb
from load_data import load_train_data, load_test_data