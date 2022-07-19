from scripts.metrics.cross_validation import cross_validation_lgbm
from scripts.metrics.mapk import mapk
from lightgbm import LGBMRanker
from scripts.utils import combine_train_sets_and_labels
import numpy as np
import warnings
import pandas as pd
from scripts.utils import create_predictions_for_second_stage, create_labels_for_second_stage
warnings.filterwarnings('ignore')

PATH_TO_TRANSACTIONS = "../data/compressed_dataset/transactions.parquet"
PATH_TO_CUSTOMERS = "../data/compressed_dataset/customers.parquet"
PATH_TO_RANKER_TRAIN_SET = '../data/ranker_train_set/'
PATH_TO_LABELS = "../data/ranker_train_labels"

df_customers = pd.read_parquet(PATH_TO_CUSTOMERS)
df_transactions = pd.read_parquet(PATH_TO_TRANSACTIONS)

model_names = [
    'lpp',
    'bivae',
    'svd'
]

lpp_best_config = {
    'num_weeks': 3
}

bivae_best_config = {
  'act_fn': 'sigmoid',
  'batch_size': 96,
  'beta_kl': 1.6324305360560756,
  'encoder_dims': [54],
  'item_frequency_threshold': 60,
  'latent_dim': 43,
  'likelihood': 'bern',
  'lr': 0.005997693826960126,
  'num_epochs': 141,
  'user_frequency_threshold': 38,
  'seed': 42,
  'gpu': True,
  'verbose': True
}

svd_best_config = {
    'bias': False,
    'iterations': 180,
    'k': 2,
    'learning_rate': 0.007659023027711605,
    'method': 'stochastic',
    'num_weeks': 7,
    'regularizer': 0.0038804778747317226,
    'verbose': False
}

print('Original LPP')
lpp_models = create_predictions_for_second_stage(
    model_name='lpp',
    config=lpp_best_config,
    transactions=df_transactions,
    all_customers=df_customers['customer_id'].unique(),
    path_to_destination_save=PATH_TO_RANKER_TRAIN_SET,
    num_train_weeks=20,
    top_k=12
)
print('Second LPP')
create_predictions_for_second_stage(
    model_name='lpp',
    config=lpp_best_config,
    transactions=df_transactions,
    all_customers=df_customers['customer_id'].unique(),
    path_to_destination_save=PATH_TO_RANKER_TRAIN_SET,
    num_train_weeks=20,
    top_k=12,
    models=lpp_models
)
print('Original BiVAE')
bivae_models = create_predictions_for_second_stage(
    model_name='bivae',
    config=bivae_best_config,
    transactions=df_transactions,
    all_customers=df_customers['customer_id'].unique(),
    path_to_destination_save=PATH_TO_RANKER_TRAIN_SET,
    num_train_weeks=20,
    top_k=12
)
print('Second BiVAE')
create_predictions_for_second_stage(
    model_name='bivae',
    config=bivae_best_config,
    transactions=df_transactions,
    all_customers=df_customers['customer_id'].unique(),
    path_to_destination_save=PATH_TO_RANKER_TRAIN_SET,
    num_train_weeks=20,
    top_k=12,
    models=bivae_models
)
print('Original SVD')
svd_models = create_predictions_for_second_stage(
    model_name='svd',
    config=svd_best_config,
    transactions=df_transactions,
    all_customers=df_customers['customer_id'].unique(),
    path_to_destination_save=PATH_TO_RANKER_TRAIN_SET,
    num_train_weeks=20,
    top_k=12
)
print('Second SVD')
create_predictions_for_second_stage(
    model_name='svd',
    config=svd_best_config,
    transactions=df_transactions,
    all_customers=df_customers['customer_id'].unique(),
    path_to_destination_save=PATH_TO_RANKER_TRAIN_SET,
    num_train_weeks=20,
    top_k=12,
    models=svd_models
)

create_labels_for_second_stage(
    transactions=df_transactions,
    path_to_destination_save=PATH_TO_LABELS,
    num_train_weeks=20,
    top_k=12
)

train_w_labels = combine_train_sets_and_labels(
    path_to_train_set=PATH_TO_RANKER_TRAIN_SET,
    path_to_train_labels=PATH_TO_LABELS,
    model_names=model_names,
    num_train_weeks=19
)

ranker = LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    eval_at=12,
    boosting_type="dart",
    max_depth=7,
    n_estimators=100,
    importance_type='gain',
    verbose=-1
  )

cv = cross_validation_lgbm(
    ranker=ranker,
    metric=mapk,
    num_folds=5,
    model_names=model_names,
    train_w_labels=train_w_labels,
    path_to_labels=PATH_TO_LABELS
  )
mean_cv = np.mean(cv)
print(cv)
print(mean_cv)
