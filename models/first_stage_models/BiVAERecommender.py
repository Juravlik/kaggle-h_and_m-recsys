from models.first_stage_models.BaseModel import BaseRecommender
import pandas as pd
import cornac
import numpy as np


class BiVAERecommender(BaseRecommender):
    def __init__(self, config: dict, cold_items_recommender=None, int_article_id=None, int_customer_id=None):
        super().__init__(cold_items_recommender, int_article_id, int_customer_id)
        self.__item_frequency_threshold = config['item_frequency_threshold']
        self.__user_frequency_threshold = config['user_frequency_threshold']
        self.__seed = config['seed']
        self.bivae = cornac.models.BiVAECF(
            k=config['latent_dim'],
            encoder_structure=config['encoder_dims'],
            act_fn=config['act_fn'],
            likelihood=config['likelihood'],
            beta_kl=config['beta_kl'],
            n_epochs=config['num_epochs'],
            batch_size=config['batch_size'],
            learning_rate=config['lr'],
            seed=config['seed'],
            use_gpu=config['gpu'],
            verbose=config['verbose']
        )

    def __preprocess_transactions(self, transactions: pd.DataFrame) -> cornac.data.Dataset:
        transactions['rating'] = 1.0
        transactions = transactions[['customer_id', 'article_id', 'rating']]
        item_counts = pd.DataFrame(transactions['customer_id'].value_counts()).reset_index()
        item_counts.columns = ['customer_id', 'num_items']

        user_counts = pd.DataFrame(transactions['article_id'].value_counts()).reset_index()
        user_counts.columns = ['article_id', 'num_users']

        item_counts = item_counts[item_counts['num_items'] >= self.__item_frequency_threshold]
        user_counts = user_counts[user_counts['num_users'] >= self.__user_frequency_threshold]
        transactions = transactions[transactions['customer_id'].isin(item_counts['customer_id'])]
        transactions = transactions[transactions['article_id'].isin(user_counts['article_id'])]
        transactions = cornac.data.Dataset.from_uir(transactions.itertuples(index=False), seed=self.__seed)
        return transactions

    def fit(self, transactions: pd.DataFrame):
        train_set = self.__preprocess_transactions(transactions)
        self.bivae.fit(train_set)

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
        items = list(self.bivae.train_set.iid_map.keys())
        for uid in customers:
            if uid in self.bivae.train_set.uid_map.keys():
                user_idx = self.bivae.train_set.uid_map[uid]
                predictions = np.sort(self.bivae.score(user_idx))[-top_k::][::-1]
                pred_args = np.argsort(self.bivae.score(user_idx))[-top_k::][::-1]
                pred_items = [items[index] for index in pred_args]
                user = [uid] * top_k
                users.extend(user)
                top_k_items.extend(pred_items)
                top_k_preds.extend(predictions.tolist())
        predict_df = pd.DataFrame(
            data={'customer_id': users, 'article_id': top_k_items, 'score': top_k_preds}
        )
        if fill_missed_customers:
            predict_df = self.get_missed_recs(predict_df, customers)

        if return_submit:
            return predict_df, self.__predict_to_submit(predict=predict_df['customer_id'].unique())
        return predict_df
