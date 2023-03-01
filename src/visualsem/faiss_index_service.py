from pathlib import Path
from typing import Dict, List, Union

import faiss
from faiss.swigfaiss import IndexIDMap
import numpy as np
import torch


class FaissIndexService:
    def __init__(self, embeddings_path: Path):
        assert embeddings_path.exists()


        self._index_suffix = ".faiss"
        self._index_factory_string = "IDMap,Flat"
        self._index_search_metric = faiss.METRIC_INNER_PRODUCT

        self._index_in_memory_cache: Dict[Path, IndexIDMap] = dict()

        self._embedding_path = embeddings_path
        self.embedding_approaches = [
            "avg_glosses_and_images",
            "avg_glosses",
            "avg_images",
        ]
        self.models = set()
        self.embeddings = {a: dict() for a in self.embedding_approaches}
        self.index_dimensions = dict()

        for emb_file in embeddings_path.glob("*.pt"):
            model = str(emb_file.name).split("_")[0]
            self.models.add(model)

            approach = None
            for a in self.embedding_approaches:
                if a in str(emb_file):
                    approach = a
                    break

            if approach is None:
                raise ValueError

            emb: Dict[str, torch.Tensor] = torch.load(emb_file)
            self.index_dimensions[model] = next(iter(emb.values())).shape[0]
            self.embeddings[approach][model] = emb
            print(f"Loaded {len(emb)} embeddings from {emb_file}")

    def _create_index(self, model: str) -> IndexIDMap:
        return faiss.index_factory(
            self.index_dimensions[model],
            self._index_factory_string,
            self._index_search_metric,
        )

    def _get_index_path_for_model_and_approach(self, model: str, approach: str):
        assert approach in self.embedding_approaches
        assert model in self.models

        return self._embedding_path / f"{model}_{approach}{self._index_suffix}"

    def _persist_index(self, index: IndexIDMap, model: str, approach: str):
        assert approach in self.embedding_approaches
        assert model in self.models

        index_fn = self._get_index_path_for_model_and_approach(
            model=model, approach=approach
        )

        if not index_fn.exists():
            if not index_fn.parent.exists():
                index_fn.parent.mkdir(parents=True, exist_ok=False)

        faiss.write_index(index, str(index_fn))
        print(f"Persisted index at {index_fn}")

    def index_exists(self, model: str, approach: str) -> bool:
        exists = self._get_index_path_for_model_and_approach(
            model=model, approach=approach
        ).exists()
        return exists

    def init_embedding_indices_for_all_models_and_approaches(self, force: bool = False):
        for approach in self.embedding_approaches:
            for model in self.models:
                index_p = self._get_index_path_for_model_and_approach(
                    model, approach)
                if index_p.exists():
                    print(
                        f"Index for model {model} and approach {approach} already exists!"
                    )
                    if force:
                        print(f"Removing index {index_p}!")
                        index_p.unlink()
                        if approach in self._index_in_memory_cache:
                            del self._index_in_memory_cache[index_p]

                    else:
                        continue

                index = self.create_or_load_index_for_model_approach(
                    model=model, approach=approach
                )

                embeddings = torch.stack(
                    [e for e in self.embeddings[approach][model].values()]
                ).numpy()
                embedding_ids = np.array(range(len(embeddings)))

                print(
                    f"Adding {len(embeddings)} embeddings to {approach} index!")
                if not len(embeddings) == len(embedding_ids):
                    raise ValueError

                # noinspection PyArgumentList
                index.add_with_ids(embeddings, embedding_ids)
                print(
                    f"Added {len(embeddings)} embeddings to Index of Approach {approach}!"
                )
                self._persist_index(
                    index=index, model=model, approach=approach)

    def create_or_load_index_for_model_approach(
        self, model: str, approach: str
    ) -> IndexIDMap:
        index_fn = self._get_index_path_for_model_and_approach(model, approach)
        index = self._index_in_memory_cache.get(index_fn, None)
        if index is not None:
            return index

        if not index_fn.exists():
            if not index_fn.parent.exists():
                index_fn.parent.mkdir(parents=True, exist_ok=False)
            print(
                f"Creating  index for model {model} and approach {approach} at {index_fn}!"
            )
            index = self._create_index(model)
            faiss.write_index(index, str(index_fn))
        else:
            # print(f"Loading  index for Approach {approach} from {index_fn} in memory!")
            index = faiss.read_index(str(index_fn))

        self._index_in_memory_cache[index_fn] = index

        return index

    def search_index(
        self, query: np.ndarray, approach: str, model: str, top_k: int = 10
    ) -> Union[Dict[int, float], List[Dict[int, float]]]:
        assert approach in self.embedding_approaches
        assert model in self.models

        if query.ndim == 1:
            query = query[np.newaxis]
        assert query.shape[1] == self.index_dimensions[model]
        assert self.index_exists(model=model, approach=approach)

        # load or create the index
        index = self.create_or_load_index_for_model_approach(
            model=model, approach=approach
        )

        if index.ntotal <= 0:
            print(f" Index for Approach {approach} is empty!")
            raise ValueError

        # noinspection PyArgumentList
        dists, ids = index.search(query, top_k)
        dists: np.ndarray = dists.squeeze()
        ids: np.ndarray = ids.squeeze()

        if ids.ndim == 1:
            return dict(zip(ids.tolist(), dists.tolist()))
        elif ids.ndim == 2:
            return [
                dict(zip(ids[i].tolist(), dists[i].tolist())) for i in range(len(ids))
            ]
        else:
            raise ValueError
