from abc import ABC, abstractmethod
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas(desc="")

class BaseRecommenderv2(ABC):
    """
    Base recommender
    """
    def __init__(
            self,
            cold_items_recommender=None,
            int_article_id=None,
            int_customer_id=None
    ):
        self._cold_items_recommender = cold_items_recommender
        self._int_article_id = int_article_id
        self._int_customer_id = int_customer_id

    def __predict_to_submit(
            self,
            predict: pd.DataFrame
    ):
        predict['article_id'] = predict['article_id'].progress_apply(lambda x: self._int_article_id[x])
        predict['customer_id'] = predict['customer_id'].progress_apply(lambda x: self._int_customer_id[x])
        predict['article_id'] = predict['article_id'].astype(str)
        predict['prediction'] = predict.groupby(['customer_id'])['article_id'].transform(lambda x: ' '.join(x))
        predict.drop_duplicates(subset=['customer_id'], inplace=True)
        predict.reset_index(inplace=True, drop=True)
        predict['prediction'] = predict['prediction'].str[:131]
        predict = predict[['customer_id', 'prediction']]

        return predict

    def get_missed_recs(self, predict_df, customers):
        cold_predictions = self._cold_items_recommender.predict(customers)
        cold_predictions = cold_predictions[~cold_predictions['customer_id'].isin(predict_df['customer_id'])]
        predict = pd.merge(predict_df, cold_predictions, on=['customer_id', 'article_id'], how='outer')
        predict = predict.fillna(-1)
        predict['score'] = predict.apply(
            lambda x: x['score_y'] if x['score_y'] != -1 else x['score_x'], axis=1)
        predict = predict.drop(columns=['score_x', 'score_y'])
        predict = predict.set_index(['customer_id']).loc[customers].reset_index()

        return predict

    @abstractmethod
    def _predict(self, customers: list,
                 top_k: int = 12) -> pd.DataFrame:
        """
        This method must return predictions
        :param customers:
        :param top_k:
        :param return_submit:
        :param fill_missed_customers:
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def fit(self, transactions: pd.DataFrame):
        pass

    def predict(self,
                customers: list,
                top_k: int = 12,
                return_submit: bool = False,
                fill_missed_customers: bool = False) -> pd.DataFrame:
        """
        This method is a wrapper for the private _predict method
        :param customers:
        :param top_k:
        :param return_submit:
        :param fill_missed_customers:
        :return:
        """
        predictions = self._predict(
            customers=customers,
            top_k=top_k)
        assert "customer_id" in predictions.columns
        assert "predicted_ids" in predictions.columns
        assert "score" in predictions.columns

        if fill_missed_customers:
            predictions = self.get_missed_recs(predictions, customers)

        if return_submit:
            return predictions, self.__predict_to_submit(predict=predictions['customer_id'].unique())
        return predictions

