from models.first_stage_models.BaseModel import BaseRecommender
import pandas as pd
from reco.recommender import FunkSVD
from collections import Counter
import numpy as np
from datetime import timedelta
from scipy.special import softmax

class SVDRecommender(BaseRecommender):
    def __init__(self, config: dict, cold_items_recommender=None, int_article_id=None, int_customer_id=None):
        super().__init__(cold_items_recommender, int_article_id, int_customer_id)
        self.__num_weeks = config['num_weeks']
        self.__verbose = config['verbose']
        self.svd = FunkSVD(
            k=config['k'],
            learning_rate=config['learning_rate'],
            regularizer=config['regularizer'],
            iterations=config['iterations'],
            method=config['method'],
            bias=config['bias']
        )

    def __get_most_freq_next_item(self, user_group):
        next_items = {}
        for user in user_group.keys():
            items = user_group[user]
            for i, item in enumerate(items[:-1]):
                if item not in next_items:
                    next_items[item] = []
                if item != items[i + 1]:
                    next_items[item].append(items[i + 1])

        pred_next = {}
        for item in next_items:
            if len(next_items[item]) >= 5:
                most_common = Counter(next_items[item]).most_common()
                ratio = most_common[0][1] / len(next_items[item])
                if ratio >= 0.1:
                    pred_next[item] = most_common[0][0]

        return pred_next

    def __preprocess_transactions(self, transactions: pd.DataFrame) -> pd.DataFrame:
        last_trans_date = transactions['t_dat'].max()
        self.__trans_weeks = []
        self.__positive_items = []
        for i in range(1, self.__num_weeks):
            selected_transactions = transactions[(transactions['t_dat'] > last_trans_date - timedelta(days=i * 7))
                                        & (transactions['t_dat'] <= last_trans_date - timedelta(days=(i - 1) * 7))].copy()
            positive_items_per_user = selected_transactions.groupby(['customer_id'])['article_id'].apply(list)
            self.__trans_weeks.append(
                selected_transactions
            )
            self.__positive_items.append(
                positive_items_per_user
            )

        train = pd.concat([self.__trans_weeks[0], self.__trans_weeks[1]], axis=0)
        train['pop_factor'] = train['t_dat'].apply(lambda x: 1 / ((last_trans_date - x).days + 1))
        popular_items_group = train.groupby(['article_id'])['pop_factor'].sum()

        _, self.__popular_items = zip(*sorted(zip(popular_items_group, popular_items_group.keys()))[::-1])
        train = pd.concat(self.__trans_weeks, axis=0)
        user_group = train.groupby(['customer_id'])['article_id'].apply(list)
        self.__pred_next = self.__get_most_freq_next_item(user_group)

        train['pop_factor'] = train['t_dat'].apply(lambda x: 1 / ((last_trans_date - x).days + 1) ** 2)
        popular_items_group = train.groupby(['article_id'])['pop_factor'].sum()

        train['feedback'] = 1
        train = train.groupby(['customer_id', 'article_id']).sum().reset_index()

        train['feedback'] = train.apply(lambda row: row['feedback'] / popular_items_group[row['article_id']], axis=1)

        train['feedback'] = train['feedback'].apply(lambda x: 5.0 if x > 5.0 else x)
        train.drop(['price', 'sales_channel_id'], axis=1, inplace=True)
        train = train.sample(frac=1).reset_index(drop=True)

        return train

    def fit(self, transactions: pd.DataFrame):
        train_set = self.__preprocess_transactions(transactions)
        self.svd.fit(
            X=train_set,
            formatizer={
                'user': 'customer_id',
                'item': 'article_id',
                'value': 'feedback'
            },
            verbose=self.__verbose
        )

    def predict(self,
        customers,
        top_k=12,
        return_submit: bool = False,
        fill_missed_customers: bool = False,
    ):
        """Computes top k predictions of recommender model from Cornac for all users data.

        Args:
            customers (list): list of customers
            top_k (int): number of recommendations
            return_submit (str): True if submission file is required
            int_customer_id (str): customer id mapping
            int_article_id (bool): items id mapping

        Returns:
            pandas.DataFrame: Dataframe with customer_id, prediction columns
        """
        users, top_k_items, top_k_preds = [], [], []
        popular_items = list(self.__popular_items)
        userindexes = {self.svd.users[i]: i for i in range(len(self.svd.users))}

        for user in customers:
            user_output = []
            predictions = []
            for positive_items_per_user in self.__positive_items:
                if user in positive_items_per_user.keys():
                    most_common_items_of_user = {k: v for k, v in Counter(positive_items_per_user[user]).most_common()}

                    user_index = userindexes[user]
                    new_order = {}
                    for k in list(most_common_items_of_user.keys())[:20]:
                        try:
                            itemindex = self.svd.items.index(k)
                            pred_value = np.dot(self.svd.userfeatures[user_index], self.svd.itemfeatures[itemindex].T) + self.svd.item_bias[
                                0, itemindex]
                        except:
                            pred_value = most_common_items_of_user[k]
                        new_order[k] = pred_value
                        predictions.append(pred_value)
                    predictions = list(softmax(predictions))
                    predictions = [1 - x for x in predictions]
                    user_output += [k for k, v in sorted(new_order.items(), key=lambda item: item[1])][:top_k]
            if len(user_output) < top_k:
                user_output += [
                    self.__pred_next[item] for item in user_output if
                    item in self.__pred_next and self.__pred_next[item] not in user_output
                ]
                user_output += list(popular_items[:top_k - len(user_output)])
            if len(predictions) < top_k:
                if len(predictions) == 0:
                    min_val = 1.0
                    predictions_pop = [min_val / (i + 1) for i in range(0, top_k - len(predictions))]
                else:
                    min_val = min(predictions)
                    predictions_pop = [min_val / (i + 2) for i in range(0, top_k - len(predictions))]
                predictions.extend(predictions_pop)
            user_output = user_output[:top_k]
            predictions = predictions[:top_k]
            user_list = [user] * top_k
            users.extend(user_list)
            top_k_items.extend(user_output)
            top_k_preds.extend(predictions)

        predict_df = pd.DataFrame(
            data={'customer_id': users, 'article_id': top_k_items, 'score': top_k_preds}
        )
        if fill_missed_customers:
            predict_df = self.get_missed_recs(predict_df, customers)

        if return_submit:
            return predict_df, self.__predict_to_submit(predict=predict_df['customer_id'].unique())
        return predict_df
