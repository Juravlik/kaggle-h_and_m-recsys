from .BiVAERecommender import BiVAERecommender
import torch
import pandas as pd
import pickle5 as pickle
from recommenders.utils.constants import SEED
from scripts.metrics.mapk import mapk
from datetime import timedelta

from .LastPurchasesPopularity import LastPurchasesPopularity
labels = pd.read_parquet('../data/ranker_train_labels/labels_1.parquet')
df_transactions = pd.read_parquet('../data/compressed_dataset/transactions.parquet')
df_transactions['t_dat'] = pd.to_datetime(df_transactions['t_dat'])
last_trans_date = df_transactions['t_dat'].max()
df_transactions = df_transactions[df_transactions['t_dat'] <= last_trans_date - timedelta(days=7)]
df_transactions = df_transactions[df_transactions['t_dat'] >= last_trans_date - timedelta(days=90)]

with open('../data/compressed_dataset/int_article_id.pickle', 'rb') as f:
    int_article_id = pickle.load(f)
with open('../data/compressed_dataset/int_customer_id.pickle', 'rb') as f:
    int_customer_id = pickle.load(f)
with open('../data/compressed_dataset/customer_id_int.pickle', 'rb') as f:
    customer_id_int = pickle.load(f)

submission = pd.read_csv('../data/sample_submission.csv')
submission['customer_id'] = submission['customer_id'].apply(lambda x: customer_id_int[x])

TOP_K = 12

config = {
    'frequency_threshold': 50,
    'latent_dim': 50,
    'encoder_dims': [100],
    'act_fn': 'tanh',
    'likelihood': 'pois',
    'num_epochs': 1,
    'batch_size': 128,
    'lr': 0.001,
    'seed': SEED,
    'gpu': torch.cuda.is_available(),
    'verbose': True
}

cold_items_recommender = LastPurchasesPopularity()
cold_items_recommender.fit(df_transactions)

model = BiVAERecommender(config, cold_items_recommender, int_article_id, int_customer_id)
# model.fit(df_transactions)
with open('./saved_models/bi_vae/bivae.pickle', 'rb') as f:
    model._bivae = pickle.load(f)
df_predict = model.predict(
    labels['customer_id'].unique(),
    top_k=TOP_K,
    fill_missed_customers=True
)
labels = labels[labels['customer_id'].isin(df_predict['customer_id'])]
map_at_k = mapk(
    labels.groupby(['customer_id'])['article_id'].apply(list).tolist(),
    df_predict.groupby(['customer_id'])['article_id'].apply(list).tolist(),
)
print(f'Success! MAP@12: {map_at_k}')

df_predict = model.predict(
    submission['customer_id'].unique(),
    top_k=TOP_K,
    fill_missed_customers=True
)
df_predict = df_predict[['customer_id', 'article_id']]
submission = model.prepare_to_submit(df_predict)
submission.to_csv('../data/output/submission.csv', index=False)
