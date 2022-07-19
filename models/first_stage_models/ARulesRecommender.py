from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
from typing import List
import pandas as pd
pd.set_option("mode.chained_assignment", "raise")

from models.first_stage_models.BaseModelv2 import BaseRecommenderv2

attribute_name_key = "attribute_name"
min_support_key = "min_support_key"
item_id_column_key = "item_id_column_key"
user_id_column_key = "user_id_column_key"
items_key = "items_key"


class ARulesRecommender(BaseRecommenderv2):
    def __init__(self,
                 config: dict,
                 cold_items_recommender=None,
                 int_article_id=None,
                 int_customer_id=None):
        """

        :param config:
        :param cold_items_recommender:
        :param int_article_id:
        :param int_customer_id:
        """
        super().__init__(cold_items_recommender=cold_items_recommender,
                         int_article_id=int_article_id,
                         int_customer_id=int_customer_id)

        self.config = config

        self.encoder = None
        self.frequent_itemsets = None
        self.rules = None
        self.previous_interactions = None

    def _rename_duplicates(self, items: List[str]):
        seen = set()
        dupes = [item for item in items if item in seen or seen.add(item)]
        renamed_items = [item + "_dup" for item in dupes]
        return list(set(items + renamed_items))

    def _interactions_to_item_lists(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """
        Create a list of items for each user
        :param interactions:
        :return:
        """
        grouped_interactions = interactions.groupby(by=[self.config[user_id_column_key]])[
            [self.config[attribute_name_key]]].agg(list).reset_index()

        grouped_interactions.loc[:, self.config[attribute_name_key]] = grouped_interactions[
            self.config[attribute_name_key]].apply(
            self._rename_duplicates)
        return grouped_interactions

    def _association_rules_analysis(self, user_item_lists: pd.DataFrame):
        encoder = TransactionEncoder()
        interaction_matrix = encoder.fit_transform(user_item_lists[self.config[attribute_name_key]])
        df = pd.DataFrame(interaction_matrix, columns=encoder.columns_)

        frequent_itemsets = fpgrowth(df,
                                     min_support=self.config[min_support_key],
                                     use_colnames=True,
                                     max_len=10)
        rules = association_rules(frequent_itemsets,
                                  metric="confidence",
                                  min_threshold=self.config[min_support_key])
        return frequent_itemsets, rules, encoder

    def _recommend_items(self, previous_user_items):
        # TODO: fix const = 30
        tmp = self.rules[self.rules['antecedents'].apply(lambda x: x.issubset(previous_user_items))][
                  "consequents"].drop_duplicates().tolist()[:20]
        result = list(set().union(*tmp))[:12]
        return result

    def _predict(self,
                customers: list,
                top_k: int = 12) -> pd.DataFrame:
        """
        Make predictions for c in customers which were  presented in train  interactions
        :param customers:
        :param top_k:
        :return:
        """
        previous_interactions = self.previous_interactions[
            self.previous_interactions[self.config[user_id_column_key]].isin(customers)]

        predictions = self._interactions_to_item_lists(previous_interactions)
        predictions["predicted_ids"] = predictions[self.config[attribute_name_key]].apply(
            self._recommend_items)
        predictions = predictions.rename(columns={self.config[item_id_column_key]: "previous_ids"})
        predictions["predicted_ids"] = predictions["predicted_ids"].apply(lambda x: [int(xi.split('_')[0]) for xi in x])
        predictions["predicted_ids"] = predictions["predicted_ids"].apply(lambda x: x[:top_k])
        predictions = predictions[predictions.predicted_ids.apply(len)>0]
        predictions["score"] = 1
        return predictions

    def fit(self, interactions: pd.DataFrame):
        interactions = interactions.copy()
        items = self.config[items_key]
        interactions.loc[:, self.config[attribute_name_key]] = interactions.loc[:, self.config[attribute_name_key]].astype(str)
        if self.config[item_id_column_key] != self.config[attribute_name_key]:
            interactions = interactions.merge(items[[self.config[item_id_column_key], self.config[attribute_name_key]]],
                                              on=self.config[item_id_column_key],
                                              how="left")
        user_item_lists = self._interactions_to_item_lists(interactions)
        frequent_itemsets, rules, encoder = self._association_rules_analysis(user_item_lists)
        self.encoder = encoder
        self.frequent_itemsets = frequent_itemsets
        self.rules = rules
        self.previous_interactions = interactions