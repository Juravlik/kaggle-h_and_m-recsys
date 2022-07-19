import pandas as pd
import os
import pickle
from datetime import datetime, timedelta


def prepare_dataset(
        path_to_customers: str,
        path_to_articles: str,
        path_to_transactions: str,
        path_to_destination_folder: str
):

    os.makedirs(path_to_destination_folder, exist_ok=True)

    ### Customers
    customers = pd.read_csv(path_to_customers, dtype={'customer_id': str})

    customer_id_int = {}
    int_customer_id = {}
    for i, cust_id in enumerate(customers['customer_id'].unique()):
        customer_id_int[cust_id] = i
        int_customer_id[i] = cust_id

    postalcode_id_int = {}
    int_postalcode_id = {}
    for i, pc in enumerate(customers['postal_code'].unique()):
        postalcode_id_int[pc] = i
        int_postalcode_id[i] = pc

    customers['customer_id'] = customers['customer_id'].apply(lambda x: customer_id_int[x])
    customers['postal_code'] = customers['postal_code'].apply(lambda x: postalcode_id_int[x])

    customers.to_parquet(os.path.join(path_to_destination_folder, 'customers.parquet'), index=False)
    del customers

    ### Articles
    articles = pd.read_csv(path_to_articles, dtype={'article_id': str})

    article_id_int = {}
    int_article_id = {}

    for i, art_id in enumerate(articles['article_id'].unique()):
        article_id_int[art_id] = i
        int_article_id[i] = art_id

    articles['article_id'] = articles['article_id'].apply(lambda x: article_id_int[x])

    articles.to_parquet(os.path.join(path_to_destination_folder, 'articles.parquet'), index=False)
    del articles

    ### Transactions
    transactions = pd.read_csv(path_to_transactions, dtype={'customer_id': str, 'article_id': str})

    transactions['customer_id'] = transactions['customer_id'].apply(lambda x: customer_id_int[x])
    transactions['article_id'] = transactions['article_id'].apply(lambda x: article_id_int[x])

    transactions.to_parquet(os.path.join(path_to_destination_folder, 'transactions.parquet'), index=False)
    del transactions

    ### Save pickles

    with open(os.path.join(path_to_destination_folder, 'customer_id_int.pickle'), 'wb') as handle:
        pickle.dump(customer_id_int, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(path_to_destination_folder, 'int_customer_id.pickle'), 'wb') as handle:
        pickle.dump(int_customer_id, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    with open(os.path.join(path_to_destination_folder, 'postalcode_id_int.pickle'), 'wb') as handle:
        pickle.dump(postalcode_id_int, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(path_to_destination_folder, 'int_postalcode_id.pickle'), 'wb') as handle:
        pickle.dump(int_postalcode_id, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(os.path.join(path_to_destination_folder, 'article_id_int.pickle'), 'wb') as handle:
        pickle.dump(article_id_int, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(path_to_destination_folder, 'int_article_id.pickle'), 'wb') as handle:
        pickle.dump(int_article_id, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_predictions_for_second_stage(
        model,
        model_name: str,
        transactions: pd.DataFrame,
        all_customers: list,
        path_to_destination_save: str,
        num_train_weeks: int = 20,
        **kwargs,
):

    os.makedirs(path_to_destination_save, exist_ok=False)

    transactions['t_dat'] = pd.to_datetime(transactions['t_dat'])
    last_trans_date = transactions['t_dat'].max()

    for i in range(num_train_weeks-1, -1, -1):

        train_set_st1 = transactions[(transactions['t_dat'] <= last_trans_date - timedelta(days=i * 7))].copy()

        if i == 0:
            candidates_predictions = all_customers

        else:
            candidates_predictions = transactions[
                (transactions['t_dat'] > last_trans_date - timedelta(days=i * 7))
                & (transactions['t_dat'] <= last_trans_date - timedelta(days=(i-1) * 7))
            ]['customer_id'].unique()

        model.fit(train_set_st1)

        df_predictions = model.predict(
            **kwargs,
            customers=candidates_predictions,
            return_submit=False,
        )

        del candidates_predictions, train_set_st1

        df_predictions['weeks_before_sub'] = i

        df_predictions.to_parquet(os.path.join(path_to_destination_save, '{}_{}.parquet'.format(model_name, i)),
                                  index=False)



def create_labels_for_second_stage(
        transactions: pd.DataFrame,
        path_to_destination_save: str,
        num_train_weeks: int = 20
):

    transactions['t_dat'] = pd.to_datetime(transactions['t_dat'])
    last_trans_date = transactions['t_dat'].max()

    for i in range(1, num_train_weeks):

        pred_set_st1 = transactions[(transactions['t_dat'] > last_trans_date - timedelta(days=i * 7))
                                    & (transactions['t_dat'] <= last_trans_date - timedelta(days=(i-1)*7))].copy()

        pred_set_st1.drop_duplicates(subset=['customer_id', 'article_id'], inplace=True)
        pred_set_st1 = pred_set_st1.groupby(['customer_id']).head(12)
        pred_set_st1['label'] = pred_set_st1.groupby(['customer_id']).cumcount()+1

        pred_set_st1['weeks_before_sub'] = i

        pred_set_st1[['customer_id', 'article_id', 'label', 'weeks_before_sub']].to_parquet(os.path.join(path_to_destination_save, 'labels_{}.parquet'.format(i)),
                                                                                            index=False)


def combine_train_sets_and_labels(
        path_to_train_set: str,
        path_to_train_labels: str,
        model_names: list,
        num_train_weeks: int = 20,
        path_to_save_result: str = None
) -> pd.DataFrame:

    for i_model, model_name in enumerate(model_names):

        for i in range(1, num_train_weeks):

            path_to_train_set_weekly = os.path.join(path_to_train_set, model_name,
                                                    '{}_{}.parquet'.format(model_name, i))

            if i == 1:
                train_set = pd.read_parquet(path_to_train_set_weekly)

            else:
                train_set = pd.concat([train_set, pd.read_parquet(path_to_train_set_weekly)], ignore_index=True)

        train_set.rename(columns={'score': 'score_{}'.format(model_name)}, inplace=True)

        if i_model == 0:
            train_set_all_models = train_set.copy()

        else:
            train_set_all_models = train_set_all_models.merge(train_set,
                                                              how='outer',
                                                              on=['customer_id', 'article_id', 'weeks_before_sub'])
        del train_set

    labels = pd.read_parquet(path_to_train_labels)

    train_set_all_models = train_set_all_models.merge(labels,
                                                      how='left',
                                                      on=['customer_id', 'article_id', 'weeks_before_sub'])

    train_set_all_models['label'] = train_set_all_models['label'].fillna(0)

    if path_to_save_result is not None:
        os.makedirs(os.path.dirname(path_to_save_result), exist_ok=True)
        train_set_all_models.to_parquet(path_to_save_result, index=False)

    return train_set_all_models


def create_one_hot_encoding(
        df: pd.DataFrame,
        feature_name: str,
        prefix: str = ''
):
    one_hot = pd.get_dummies(df[feature_name], prefix=prefix)

    df = df.drop(feature_name, axis=1)
    df = df.join(one_hot)

    return df


if __name__ == "__main__":
    prepare_dataset(
        path_to_customers='/home/juravlik/PycharmProjects/kaggle_hnm_recsys/data/initial_dataset/customers.csv',
        path_to_articles='/home/juravlik/PycharmProjects/kaggle_hnm_recsys/data/initial_dataset/articles.csv',
        path_to_transactions='/home/juravlik/PycharmProjects/kaggle_hnm_recsys/data/initial_dataset/transactions_train.csv',
        path_to_destination_folder='/home/juravlik/PycharmProjects/kaggle_hnm_recsys/data/compressed_dataset'
    )
