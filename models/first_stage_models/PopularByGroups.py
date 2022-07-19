from models.first_stage_models.BaseModel import BaseRecommender
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


class PopularByGroupsRecommender(BaseRecommender):

    def __init__(
            self,
            config: dict,
            int_article_id: dict = None,
            int_customer_id: dict = None
    ):
        super().__init__(
            config=config,
            cold_items_recommender=None,
            int_article_id=int_article_id,
            int_customer_id=int_customer_id
        )

        self.int_article_id = int_article_id
        self.int_customer_id = int_customer_id
        self.num_last_days_for_popularity = config['num_last_days_for_popularity']
        self.groups = config['groups']

        self.customers_by_group = None
        self.popular_by_group = None
        self.scores_by_group = None

    def fit(
            self,
            transactions: pd.DataFrame,
            customers_with_groups: pd.DataFrame
    ):

        transactions['t_dat'] = pd.to_datetime(transactions['t_dat'])
        last_trans_date = transactions['t_dat'].max()

        transactions = transactions[
            transactions['t_dat'] >= last_trans_date - timedelta(days=self.num_last_days_for_popularity - 1)]

        self.customers_by_group = []
        self.popular_by_group = []
        self.scores_by_group = []

        for group, df in customers_with_groups.groupby(self.groups):

            cur_customers = df['customer_id'].tolist()
            cur_trans = transactions[transactions['customer_id'].isin(cur_customers)]

            if cur_trans.shape[0] == 0:
                continue

            self.customers_by_group.append(cur_customers)
            self.popular_by_group.append(list(cur_trans['article_id'].value_counts().index))
            self.scores_by_group.append(list(cur_trans['article_id'].value_counts() /
                                             cur_trans['article_id'].value_counts().iloc[0]))

    def predict(
            self,
            customers: list,
            top_k: int = 12,
            return_submit: bool = False,
            fill_missed_customers: bool = False
    ):

        prediction_list = []
        customer_list = []
        score_list = []

        for i in range(len(self.customers_by_group)):
            for customer in list(set(customers) & set(self.customers_by_group[i])):
                prediction_list += self.popular_by_group[i][:top_k]
                score_list += self.scores_by_group[i][:top_k]
                customer_list += [customer] * len(self.popular_by_group[i][:top_k])

        predict_df = pd.DataFrame({'customer_id': customer_list,
                                   'article_id': prediction_list,
                                   'score': score_list})

        if return_submit:
            return predict_df, super().predict_to_submit(predict=predict_df.copy())

        else:
            return predict_df
