from pathlib import Path
from typing import Callable, Dict, List, Union, Tuple

import faiss
from faiss.swigfaiss import IndexIDMap
from loguru import logger
from matplotlib.pyplot import text
import numpy as np
import torch
import torch.nn.functional as F

from src.model.clip_encoder import ICLIPEncoder
from src.vwsd_dataset import VWSDSampleBase
from src.visualsem.visualsem_dataset_nodes import VisualSemNodesDataset


class VisualSemClient:
    def __init__(
        self,
        clip_encoder: ICLIPEncoder,
        vs: VisualSemNodesDataset,
        vs_embeddings_path: Path,
        langs: List[str] = ["en"],
        normalize_embeddings: bool = True,
        init_all: bool = True,
        force_init: bool = False,
        get_sample_text: Callable[[VWSDSampleBase], str] | None = None,
    ):
        assert (
            vs_embeddings_path.exists()
        ), f"Cannot read VisualSem Embeddings at {vs_embeddings_path}"

        self.vs = vs

        self.encoder = clip_encoder
        self.encoder_dimension = (
            self.encoder.encode_text(["get hidden dim"]).squeeze().shape[0]
        )

        self._index_suffix = ".faiss"
        self._index_factory_string = "IDMap,Flat"
        self._index_search_metric = faiss.METRIC_INNER_PRODUCT

        self.normalize_embeddings = normalize_embeddings

        self.vs_embeddings_path = vs_embeddings_path
        self.emb_2_node_id: Dict[str, Dict[str, Dict[int, int]]] = dict()
        self.node_2_emb_id: Dict[str, Dict[str, Dict[int, List[int]]]] = dict()

        # Dict[lang: str, Dict[approach: str, Dict[bnid: str, Tensor]]]
        self.vs_embeddings, self.vs_embedding_dimension = self._load_vs_embeddings(
            langs
        )

        assert (
            self.encoder_dimension == self.vs_embedding_dimension
        ), f"Encoder Dimensions {self.encoder_dimension} do not match VisualSem Embeddings Dimensions {self.vs_embedding_dimension}!"

        self.vs_embedding_langs = set(self.vs_embeddings.keys())

        # Dict[lang: str, Dict[approach: str, faiss_index: IndexIDMap]]
        self._index_in_memory_cache: Dict[str, Dict[str, IndexIDMap]] = dict()
        if init_all:
            self.__init_indices_from_embeddings(force_init)

        if get_sample_text is None:
            get_sample_text = (
                lambda s: f'A description for "{s.word}" as in "{s.context}"'
            )
        self.get_sample_text = get_sample_text

    def _load_vs_embeddings(
        self,
        langs: List[str],
    ) -> Tuple[Dict[str, Dict[str, Dict[str, torch.Tensor]]], int]:
        embeddings = dict()
        dim = -1

        for emb_p in self.vs_embeddings_path.glob(
            f"{self.encoder.model_name.replace('/', '-')}*.pt"
        ):
            approach = "_".join(emb_p.stem.split("_")[1:3])
            # if not approach.startswith('avg'):
            #     logger.warning(f"Skipping VisualSem Embeddings {emb_p}")
            #     continue
            gloss_lang = emb_p.stem.split("_")[-1]
            if gloss_lang not in langs:
                continue

            embs = torch.load(emb_p)

            # to map between embedding id to node id if a node has multiple embeddings (single approach)
            if gloss_lang not in self.emb_2_node_id:
                self.emb_2_node_id[gloss_lang] = dict()
                self.node_2_emb_id[gloss_lang] = dict()
            if approach not in self.emb_2_node_id[gloss_lang]:
                self.emb_2_node_id[gloss_lang][approach] = dict()
                self.node_2_emb_id[gloss_lang][approach] = dict()
            current_emb_id = 0
            for node_id, emb in enumerate(embs.values()):
                self.node_2_emb_id[gloss_lang][approach][node_id] = []
                if emb.ndim == 2:
                    for _ in emb:
                        self.emb_2_node_id[gloss_lang][approach][
                            current_emb_id
                        ] = node_id
                        self.node_2_emb_id[gloss_lang][approach][node_id].append(current_emb_id)
                        current_emb_id += 1
                elif emb.ndim == 1:
                    self.emb_2_node_id[gloss_lang][approach][current_emb_id] = node_id
                    self.node_2_emb_id[gloss_lang][approach][node_id].append(current_emb_id)
                    current_emb_id += 1

            # assert all embeddings have the same hidden dim
            emb = next(iter(embs.values()))
            hidden_dim = emb.shape[-1]
            if dim == -1:
                dim = hidden_dim
            assert (
                hidden_dim == dim
            ), f"Embedding Dimensions {(dim, hidden_dim)} do not match!"

            logger.info(
                f"Loaded  {len(embs)} VisualSem {approach} Node Embeddings from {emb_p}"
            )
            if gloss_lang not in embeddings:
                embeddings[gloss_lang] = dict()

            embeddings[gloss_lang][approach] = embs

        return embeddings, dim

    def get_node_embedding(self, node_id: int, lang: str, approach: str) -> torch.Tensor:
        bnid = self.vs[node_id]["bnid"]
        emb = self.vs_embeddings[lang][approach][
            bnid
        ].to(self.encoder.device)
        return emb

    def _create_empty_index(self) -> IndexIDMap:
        return faiss.index_factory(
            self.encoder_dimension,
            self._index_factory_string,
            self._index_search_metric,
        )

    def _assert_lang_and_approach_exist(self, lang: str, approach: str) -> None:
        assert (
            lang in self.vs_embedding_langs
        ), f"Language {lang} not available! Supported langs: {self.vs_embedding_langs}"
        assert (
            approach in self.vs_embeddings[lang]
        ), f"Approach {approach} not available! Supported: {set(self.vs_embeddings[lang].keys())}"

    def _get_index_path_for_lang_and_approach(self, lang: str, approach: str) -> Path:
        self._assert_lang_and_approach_exist(lang, approach)
        return (
            self.vs_embeddings_path
            / f"{self.encoder.model_name.replace('/', '-')}_{approach}_{lang}{self._index_suffix}"
        )

    def _persist_index(self, lang: str, approach: str):
        self._assert_lang_and_approach_exist(lang, approach)

        index_fn = self._get_index_path_for_lang_and_approach(
            lang=lang, approach=approach
        )

        if not index_fn.exists():
            if not index_fn.parent.exists():
                index_fn.parent.mkdir(parents=True, exist_ok=False)

        index = self._index_in_memory_cache[lang][approach]
        faiss.write_index(index, str(index_fn))
        logger.info(f"Persisted Index at {index_fn}")

    def _index_exists(self, lang: str, approach: str) -> bool:
        exists = self._get_index_path_for_lang_and_approach(
            lang=lang, approach=approach
        ).exists()
        return exists

    def _init_index_from_vs_embeddings(
        self, lang: str, approach: str, force: bool
    ) -> None:
        index_p = self._get_index_path_for_lang_and_approach(lang, approach)
        if index_p.exists():
            logger.info(
                f"Index for language {lang} and approach {approach} already exists!"
            )
            if force:
                logger.warning(f"Removing index {index_p}!")
                index_p.unlink()
                if lang in self._index_in_memory_cache:
                    if approach in self._index_in_memory_cache[lang]:
                        del self._index_in_memory_cache[lang][approach]
            else:
                return

        embeddings = []
        for i, embs in enumerate(self.vs_embeddings[lang][approach].values()):
            if embs.ndim == 2:
                for j, emb in enumerate(embs):
                    embeddings.append(emb)
            elif embs.ndim == 1:
                embeddings.append(embs)
        embeddings = torch.stack(embeddings)
        assert len(self.emb_2_node_id[lang][approach]) == embeddings.shape[0]

        if self.normalize_embeddings:
            embeddings = F.normalize(embeddings, dim=1)
        embeddings = embeddings.numpy()
        embedding_ids = np.array(range(len(embeddings)))

        if not len(embeddings) == len(embedding_ids):
            raise ValueError

        # create an empty index and add the embeddings
        logger.info(
            f"Adding {len(embeddings)} embeddings to {approach} {lang} index..."
        )
        index = self._create_empty_index()
        index.add_with_ids(embeddings, embedding_ids)  # type: ignore
        logger.info(
            f"Adding {len(embeddings)} embeddings to {approach} {lang} index... Done!"
        )

        # store the index in memory and persist on disk
        if lang not in self._index_in_memory_cache:
            self._index_in_memory_cache[lang] = dict()
        self._index_in_memory_cache[lang][approach] = index
        self._persist_index(lang=lang, approach=approach)

    def __init_indices_from_embeddings(self, force: bool = False):
        for lang in self.vs_embeddings.keys():
            for approach in self.vs_embeddings[lang].keys():
                self._init_index_from_vs_embeddings(lang, approach, force)

    def _load_index_from_disk(self, lang: str, approach: str) -> IndexIDMap:
        index_fn = self._get_index_path_for_lang_and_approach(lang, approach)
        assert index_fn.exists(), f"Cannot read Index at {index_fn}"

        index = faiss.read_index(str(index_fn))
        # store the index in memory and persist on disk
        if lang not in self._index_in_memory_cache:
            self._index_in_memory_cache[lang] = dict()
        self._index_in_memory_cache[lang][approach] = index
        return index

    def _get_index(self, lang: str, approach: str) -> IndexIDMap | None:
        # get the index from memory or disk if it exists
        if lang in self._index_in_memory_cache:
            if approach in self._index_in_memory_cache[lang]:
                return self._index_in_memory_cache[lang][approach]
        elif self._index_exists(lang, approach):
            return self._load_index_from_disk(lang, approach)

        return None

    def encode_query(
        self,
        query: VWSDSampleBase
        | List[VWSDSampleBase]
        | str
        | List[str]
        | List[torch.Tensor]
        | List[List[torch.Tensor]],
        text_or_images: str = "text",
    ) -> np.ndarray:
        if text_or_images == "text":
            if isinstance(query, VWSDSampleBase):
                q = [self.get_sample_text(query)]
            elif isinstance(query, str):
                q = [query]
            elif isinstance(query, List) and isinstance(query[0], VWSDSampleBase):
                q = [self.get_sample_text(s) for s in query]
            elif isinstance(query, List) and isinstance(query[0], str):
                q = query
            else:
                raise ValueError
            query_embedding = (
                self.encoder.encode_text(q, self.normalize_embeddings).cpu().numpy()
            )
        elif text_or_images == "images":
            if isinstance(query, VWSDSampleBase):
                q = query.images
            elif isinstance(query, List) and isinstance(query[0], torch.Tensor):
                q = query
            elif isinstance(query, List) and isinstance(query[0], VWSDSampleBase):
                q = [s.images for s in query]
            elif (
                isinstance(query, List)
                and isinstance(query[0], List)
                and isinstance(query[0][0], torch.Tensor)
            ):
                q = query
            else:
                raise ValueError
            query_embedding = (
                self.encoder.encode_images(q, self.normalize_embeddings).cpu().numpy()
            )
        else:
            raise ValueError

        if query_embedding.ndim == 1:
            query_embedding = query_embedding[np.newaxis]
        assert query_embedding.shape[1] == self.vs_embedding_dimension

        return query_embedding

    def get_index(
        self, vs_embeddings_approach: str = "avg_glosses", vs_gloss_lang: str = "en"
    ) -> IndexIDMap:
        index = self._get_index(vs_gloss_lang, vs_embeddings_approach)
        if index is None or index.ntotal <= 0:  # type: ignore
            self._init_index_from_vs_embeddings(
                vs_gloss_lang, vs_embeddings_approach, True
            )
            index = self._get_index(vs_gloss_lang, vs_embeddings_approach)
        return index

    def get_best_matching_visual_sem_nodes(
        self,
        query: VWSDSampleBase
        | List[VWSDSampleBase]
        | str
        | List[str]
        | List[torch.Tensor]
        | List[List[torch.Tensor]],
        text_or_images: str = "text",
        vs_embeddings_approach: str = "avg_glosses",
        vs_gloss_lang: str = "en",
        top_k: int = 10,
    ) -> List[Tuple[int, float]] | List[List[Tuple[int, float]]]:
        self._assert_lang_and_approach_exist(vs_gloss_lang, vs_embeddings_approach)

        # get the index for the language and embedding approach
        index = self.get_index(vs_embeddings_approach, vs_gloss_lang)

        # encode sample(s)
        query_embedding = self.encode_query(query, text_or_images)

        # noinspection PyArgumentList
        emb_dists, emb_ids = index.search(query_embedding, 100)

        results: List[Tuple[int, float]] = list()
        for ids, dists in zip(emb_ids, emb_dists):
            per_node_accumulated_dists = dict()
            for emb_id, dist in zip(ids, dists):
                nid = self.emb_2_node_id[vs_gloss_lang][vs_embeddings_approach][emb_id]
                if nid not in per_node_accumulated_dists:
                    per_node_accumulated_dists[nid] = 0
                per_node_accumulated_dists[nid] += dist
            per_node_accumulated_dists: Dict[int, float] = {
                k: v
                for k, v in sorted(
                    per_node_accumulated_dists.items(),
                    key=lambda i: i[1],
                    reverse=True,
                )
            }
            top_k_nids = list(per_node_accumulated_dists.items())[:top_k]
            results.append(top_k_nids)

        if top_k == 1:
            return list(map(lambda r: r[0], results))
        elif emb_ids.ndim == 2:
            return results
        else:
            raise ValueError
