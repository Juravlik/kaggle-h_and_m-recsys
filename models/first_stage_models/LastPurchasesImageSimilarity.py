import pandas as pd
from scripts.images_scripts.similarity_search import SimilaritySearch
from models.first_stage_models.BaseModel import BaseRecommender


class LastPurchasesImageSimilarity(BaseRecommender):
    def __init__(
            self,
            searcher: SimilaritySearch,
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

        self.searcher = searcher
        self.num_last_purhases = config['num_last_purchases']
        self.num_similar_articles_per_each_purchase = config['num_similar_articles_per_each_purchase']
        self.int_article_id = int_article_id
        self.int_customer_id = int_customer_id

        self.transactions = None

    def fit(
            self,
            transactions: pd.DataFrame
    ):

        transactions['t_dat'] = pd.to_datetime(transactions['t_dat'])

        transactions = transactions.sort_values(['t_dat'], ascending=False)
        transactions = transactions.groupby(['customer_id', 'article_id']).first().reset_index()
        transactions = transactions.sort_values(['t_dat'], ascending=False)

        self.transactions = transactions.groupby(['customer_id']).head(self.num_last_purhases)


    def predict(
            self,
            customers: list,
            return_submit: bool = False
    ):

        self.transactions = self.transactions[self.transactions['customer_id'].isin(customers)]
        prediction_list = []
        customer_list = []
        score_list = []

        for cust_id, art_id in zip(self.transactions['customer_id'].tolist(),
                                   self.transactions['article_id'].tolist()):

            scores, similar_items = self.searcher.search_similar(
                target_int_article_id=art_id,

                # +1 because first element is a query_item (art_id)
                n_images=self.num_similar_articles_per_each_purchase + 1
            )

            if similar_items is None:
                continue

            # drop first element because first element is a query_item
            score_list += list(scores)[1:]
            prediction_list += list(similar_items)[1:]
            customer_list += [cust_id] * len(list(scores)[1:])

        score_list = [s/2 + 0.5 for s in score_list]

        predict_df = pd.DataFrame({'customer_id': customer_list,
                                   'article_id': prediction_list,
                                   'score': score_list})

        predict_df = predict_df.groupby(['customer_id', 'article_id'])['score'].max().reset_index()

        if return_submit:
            return predict_df, super().predict_to_submit(predict=predict_df.copy())

        else:
            return predict_df
