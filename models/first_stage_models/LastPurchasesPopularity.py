import pandas as pd
from models.first_stage_models.BaseModel import BaseRecommender
from datetime import timedelta


class LastPurchasesPopularity(BaseRecommender):
    def __init__(self, config: dict, int_article_id=None, int_customer_id=None):
        super().__init__(
            cold_items_recommender=None,
            int_article_id=int_article_id,
            int_customer_id=int_customer_id
        )
        self.num_weeks = config['num_weeks']
        self.num_last_weeks_for_purchases = config['num_last_weeks_for_purchases']

    def fit(
            self,
            transactions: pd.DataFrame
    ):
        transactions['t_dat'] = pd.to_datetime(transactions['t_dat'])
        last_trans_date = transactions['t_dat'].max()
        self.trans_weeks = []

        for i in range(self.num_last_weeks_for_purchases):
            self.trans_weeks.append(
                transactions[transactions['t_dat'] >= last_trans_date - timedelta(days=(1 + i) * 7)].copy())

        self.purchase_dicts = [{}] * self.num_last_weeks_for_purchases
        self.dummy_lists = []

        for i_week in range(self.num_last_weeks_for_purchases):
            for i, x in enumerate(zip(self.trans_weeks[i_week]['customer_id'], self.trans_weeks[i_week]['article_id'])):
                cust_id, art_id = x
                if cust_id not in self.purchase_dicts[i_week]:
                    self.purchase_dicts[i_week][cust_id] = {}

                if art_id not in self.purchase_dicts[i_week][cust_id]:
                    self.purchase_dicts[i_week][cust_id][art_id] = 0

                self.purchase_dicts[i_week][cust_id][art_id] += 1

            self.dummy_lists.append(list((self.trans_weeks[i_week]['article_id'].value_counts()).index))

    def predict(
            self,
            customers: list,
            top_k: int = 12,
            return_submit: bool = False
    ):

        predict = pd.DataFrame({'customer_id': customers})[['customer_id']]

        prediction_list = []
        customer_list = []
        score_list = []

        dummy_pred = []

        for pred in self.dummy_lists[0]:
            if pred not in dummy_pred:
                dummy_pred += [pred]
            if len(dummy_pred) >= top_k:
                break

        for i, cust_id in enumerate(predict['customer_id'].values.reshape((-1,))):

            last_week_purchase = 0
            while last_week_purchase < len(self.purchase_dicts) and cust_id not in self.purchase_dicts[last_week_purchase]:
                last_week_purchase += 1

            if last_week_purchase < len(self.purchase_dicts):
                l = sorted((self.purchase_dicts[last_week_purchase][cust_id]).items(), key=lambda x: x[1], reverse=True)
                unique_l = []

                for y in l:
                    if y[0] not in unique_l:
                        unique_l += [y[0]]
                l = unique_l

                if len(l) > top_k:
                    s = l[:top_k]

                else:
                    s = l
                    for dum in self.dummy_lists[last_week_purchase]:
                        if len(s) >= top_k:
                            break
                        else:
                            if dum not in s:
                                s += [dum]

            else:
                s = dummy_pred

            for j in range(len(s)):
                score_list += [1 / (j + 1)]

            prediction_list += s
            customer_list += [cust_id] * len(s)

        predict_df = pd.DataFrame({'customer_id': customer_list,
                                   'article_id': prediction_list,
                                   'score': score_list})

        if return_submit:
            return predict_df, super().predict_to_submit(predict=predict_df.copy())

        else:
            return predict_df

    def predict_for_selected_pairs(
            self,
            customers: list,
            articles: list
    ):

        score_list = []

        dummy_pred = self.dummy_lists[0]

        i = 0
        for cust_id, art_id in zip(customers, articles):

            i += 1

            last_week_purchase = 0
            while last_week_purchase < len(self.purchase_dicts) and cust_id not in self.purchase_dicts[
                last_week_purchase]:
                last_week_purchase += 1

            if last_week_purchase < len(self.purchase_dicts):
                l = sorted((self.purchase_dicts[last_week_purchase][cust_id]).items(), key=lambda x: x[1], reverse=True)
                l = [y[0] for y in l]

                s = l + self.dummy_lists[last_week_purchase]

            else:
                s = dummy_pred

            j = 0

            while j < len(s) and art_id != s[j]:
                j += 1

            if j == len(s):
                score = 0
            else:
                score = 1 / (j + 1)

            score_list.append(score)

        predict_df = pd.DataFrame({'customer_id': customers,
                                   'article_id': articles,
                                   'score': score_list})

        return predict_df
