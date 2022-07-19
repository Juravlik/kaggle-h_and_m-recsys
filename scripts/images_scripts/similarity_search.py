from scripts.images_scripts.index import Index
import torch
import numpy as np
import pandas as pd


class SimilaritySearch:
    def __init__(self, index: Index,
                 parquet_file_with_articles_index: str, pickle_file_with_embeddings: str,
                 path_article_id_int: str,
                 ):
        self.index = index

        self.article_index = pd.read_parquet(parquet_file_with_articles_index)
        self.embeddings = pd.read_pickle(pickle_file_with_embeddings)
        self.article_id_int = pd.read_pickle(path_article_id_int)

        self.article_index['article_id'] = self.article_index['article_id'].apply(lambda x: self.article_id_int[x])

    def reset(self):
        self.index.reset()

    def _get_article_id_from_indexes(self, indexes) -> np.array:
        return self.article_index.iloc[indexes]['article_id'].values

    def search_similar(self, target_int_article_id: int, n_images: int):

        if target_int_article_id not in self.embeddings:
            return None, None

        embedding = self.embeddings[target_int_article_id]

        dists, indexes = self.index.predict(np.array([embedding]), n_images)

        return dists.flatten(), np.apply_along_axis(self._get_article_id_from_indexes, 1, indexes).flatten()


if __name__ == '__main__':
    pass
