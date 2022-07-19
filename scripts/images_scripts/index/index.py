import os
import torch
import numpy as np
import faiss
import pickle


class Index:
    def __init__(
            self,
            dimension: int,
            device=torch.device('cpu')
    ):
        # metric is a list of numbers for recall@k

        # vectors need to create without shuffle
        self.dimension = dimension
        self.device = device
        self.index = None
        self.embeddings = list()

    @staticmethod
    def l2_normalize(v) -> np.array:
        if len(v.shape) == 1:
            # if only one vector
            norm = np.linalg.norm(v)
            return np.asarray(v) / norm
        elif len(v.shape) == 2:
            # if v == matrix
            return v / np.expand_dims(np.linalg.norm(v, axis=1, ord=2), axis=1)

    def build_index(self):
        pass

    def reset(self):
        pass

    def predict(self, embedding: np.array, n_images: int):
        pass

    def add_batch(self, embeddings_batch):
        self.embeddings.append(embeddings_batch)

    def _index_to_gpu(self):
        pass

    def _index_to_cpu(self):
        pass

    def save_ranking_model(self, path_to_save_ranking_model_path: str):
        with open(path_to_save_ranking_model_path, "wb+") as f:
            pickle.dump(self.index, f)

    def load_ranking_model(self, path_to_load_model):
        with open(path_to_load_model, 'rb') as f:
            self.index = pickle.load(f)


class AbstractFaissIndex(Index):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._create_index()

        if self.device.type == 'cuda':
            self._index_to_gpu()

    def _create_index(self):
        pass

    def _index_to_gpu(self):
        gpu_resources = faiss.StandardGpuResources()
        self.index = faiss.index_cpu_to_gpu(gpu_resources, self.device.index or 0, self.index)

    def _index_to_cpu(self):
        self.index = faiss.index_gpu_to_cpu(self.index)

    def build_index(self):
        all_embeddings = np.concatenate(self.embeddings).astype('float32')
        all_embeddings = self.l2_normalize(all_embeddings)
        self.index.train(all_embeddings)
        self.index.add(all_embeddings)

    def predict(self, embedding, n_images):
        embedding = self.l2_normalize(embedding)
        embedding = embedding.astype('float32')
        dists, indexes = self.index.search(embedding, n_images)

        return dists, indexes

    def reset(self):
        self.embeddings = list()
        self.index.reset()
        #self.index = None

    def save_ranking_model(self, path_to_save_ranking_model_path):
        if self.device.type == 'cuda':
            self._index_to_cpu()

        path = os.path.join(path_to_save_ranking_model_path, "faiss.index")
        faiss.write_index(self.index, path)

    def load_ranking_model(self, path_to_load_model):
        self.index = faiss.read_index(path_to_load_model)


class IVFFaissIndex(AbstractFaissIndex):
    def __init__(
            self,
            n_list: int = None,
            n_probe: int = None,
            **kwargs,
    ):
        self.n_list = n_list
        self.n_probe = n_probe

        super().__init__(**kwargs)

    def _create_index(self):
        quantizer = faiss.IndexFlatIP(self.dimension)
        self.index = faiss.IndexIVFFlat(quantizer, self.dimension, self.n_list, faiss.METRIC_INNER_PRODUCT)
        self.index.nprobe = self.n_probe


class FlatFaissIndex(AbstractFaissIndex):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _create_index(self):
        self.index = faiss.IndexFlatIP(self.dimension)


if __name__ == '__main__':
    # batch_add_1 = np.array([[1, 2, 3, 4], [4, 4, 4, 4], [2, 1, 6, 7]])
    # batch_add_2 = np.array([[2, 2, 4, 5], [9, 4, 2, 1], [3, 4, 5, 6]])
    # batch_add_3 =np.array([[4, 5, 6, 7]])
    #
    # index = FlatFaissIndex(dimension=4, device=torch.device('cpu'))
    # index.add_batch(batch_add_1)
    # index.add_batch(batch_add_2)
    # index.add_batch(batch_add_3)
    # index.build_index()
    #
    # # batch_query_1 = np.array([[1, 2, 3, 3], [3, 4, 4, 5], [2, 1, 6, 8]])
    # # batch_query_1 = np.array([[1, 2, 3, 5]])
    # print(index.predict(batch_query_1, 3))
    #
    # index.reset()

    index = FlatFaissIndex(dimension=4)

    array = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    index.add_batch(array)
    index.build_index()
    index.reset()
    new_array = np.array([[9, 10, 11, 12], [13, 14, 15, 16]])
    index.add_batch(new_array)
    index.build_index()
    print(index.predict(np.array([[1, 2, 3, 4]]), 2))
    print()
