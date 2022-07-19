import pandas as pd
import os
from lightgbm import LGBMRanker
from datetime import timedelta
from models.first_stage_models.LastPurchasesPopularity import LastPurchasesPopularity
from models.first_stage_models.BiVAERecommender import BiVAERecommender
from models.first_stage_models.SVDRecommender import SVDRecommender


def cross_validation(
        model_name,
        config,
        metric,
        num_folds,
        path_to_transactions: str,
        path_to_labels_folder: str,
) -> list:
    transactions = pd.read_parquet(path_to_transactions)
    transactions['t_dat'] = pd.to_datetime(transactions['t_dat'])
    last_trans_date = transactions['t_dat'].max()
    results = []
    for i in range(num_folds):
        labels = pd.read_parquet(os.path.join(path_to_labels_folder, 'labels_{}.parquet'.format(i + 1)))
        if model_name == 'lpp':
            model = LastPurchasesPopularity(config)
        elif model_name == 'bivae':
            model = BiVAERecommender(config)
        elif model_name == 'svd':
            model = SVDRecommender(config)
        model.fit(transactions[transactions['t_dat'] <= last_trans_date - timedelta(days=(i + 1) * 7)])
        predicts = model.predict(labels['customer_id'].unique(), return_submit=False)
        results.append(
            metric(
                labels.groupby(['customer_id'])['article_id'].apply(list).tolist(),
                predicts.groupby(['customer_id'])['article_id'].apply(list).tolist(),
                k=12
            )
        )
    return results[::-1]


def cross_validation_lgbm(
        ranker_config,
        metric,
        num_folds,
        model_names,
        train_w_labels,
        path_to_labels
) -> list:
    score_columns = [f'score_{x}' for x in model_names]
    weeks = sorted(train_w_labels['weeks_before_sub'].unique())
    results = []
    for i in range(0, num_folds):
        path_to_fold_labels = os.path.join(path_to_labels, f'labels_{i+1}.parquet')
        labels = pd.read_parquet(path_to_fold_labels)
        last_week = train_w_labels[train_w_labels['weeks_before_sub'] == weeks[i]]
        train_w_labels = train_w_labels[train_w_labels['weeks_before_sub'] > weeks[i]]
        ranker = LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            eval_at=12,
            boosting_type=ranker_config['boosting_type'],
            max_depth=ranker_config['max_depth'],
            n_estimators=ranker_config['n_estimators'],
            importance_type=ranker_config['importance_type'],
            verbose=-1
        )
        ranker = ranker.fit(
            X=train_w_labels[score_columns],
            y=train_w_labels[['label']],
            group=train_w_labels.groupby(['weeks_before_sub', 'customer_id'])['article_id'].count().values
        )

        last_week['predict'] = ranker.predict(last_week[score_columns])
        last_week = last_week.sort_values(['customer_id', 'predict'], ascending=False).groupby('customer_id').head(12)

        results.append(
            metric(
                labels.groupby(['customer_id'])['article_id'].apply(list).tolist(),
                last_week.groupby(['customer_id'])['article_id'].apply(list).tolist(),
                k=12
            )
        )

    return results[::-1]
