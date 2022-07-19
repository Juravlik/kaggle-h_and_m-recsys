from BiVAERecommender import BiVAERecommender
import pandas as pd
from scripts.metrics.mapk import mapk
from datetime import timedelta
import warnings
from scripts.utils import create_labels_for_second_stage
warnings.filterwarnings('ignore')

TOP_K = 120

df_transactions = pd.read_parquet('../../data/compressed_dataset/transactions.parquet')
create_labels_for_second_stage(df_transactions, '../../data/ranker_train_labels', top_k=TOP_K)
labels = pd.read_parquet('../../data/ranker_train_labels/labels_1.parquet')
df_transactions['t_dat'] = pd.to_datetime(df_transactions['t_dat'])
last_trans_date = df_transactions['t_dat'].max()
df_transactions = df_transactions[df_transactions['t_dat'] <= last_trans_date - timedelta(days=7)]

config = {
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
     'gpu': False,
     'verbose': False
}

model = BiVAERecommender(config)
model.fit(df_transactions)
df_predict = model.predict(
    labels['customer_id'].unique(),
    top_k=TOP_K,
    fill_missed_customers=False
)
map_at_k = mapk(
    labels.groupby(['customer_id'])['article_id'].apply(list).tolist(),
    df_predict.groupby(['customer_id'])['article_id'].apply(list).tolist(),
    TOP_K
)
print(float(map_at_k))
# 0.00543778607955383
