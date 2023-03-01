from typing import List, Dict, Tuple

import torch.nn.functional as F
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np

from src.algo.algo import Algorithm
from src.model.clip_encoder import ICLIPEncoder
from src.visual_sem_client import VisualSemClient
from src.vwsd_dataset import VWSDDataset, VWSDSampleBase
from tqdm import tqdm
from loguru import logger


class VisualSemTextFirst(Algorithm):
    def __init__(
        self,
        encoder: ICLIPEncoder,
        ds: VWSDDataset,
        vsc: VisualSemClient,
        vs_gloss_lang: str = "en",
        t2i: bool = False,
        single_embs: bool = False,
        i2i_single_embs: bool = False,
        normalize: bool = True,
        show_progress_bar: bool = True,
    ):
        super().__init__(
            f"VisualSemTextFirst({'t2i' if t2i else 't2t'}-{'SingleEmbs' if single_embs else 'AvgEmbs'},i2i-{'SingleEmbs' if i2i_single_embs else 'AvgEmbs'})",
            encoder,
            ds,
            normalize,
            show_progress_bar,
        )
        self.vsc = vsc
        self.vs_gloss_lang = vs_gloss_lang
        self.single_embs = single_embs
        self.t2i = t2i

        if self.t2i and self.single_embs:
            self.vs_embeddings_approach = "single_images"
        elif self.t2i and not self.single_embs:
            self.vs_embeddings_approach = "avg_images"
        elif not self.t2i and self.single_embs:
            self.vs_embeddings_approach = "single_glosses"
        elif not self.t2i and not self.single_embs:
            self.vs_embeddings_approach = "avg_glosses"

        self.i2i_single_embs = i2i_single_embs

    def get_sample_text(self, sample: VWSDSampleBase) -> str | List[str]:
        return self.vsc.get_sample_text(sample)

    def _single_compute_i2i_sims_avg_images(
        self, sample: VWSDSampleBase, nid2score: List[Tuple[int, float]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        i_repr = self.compute_visual_repr(self.get_sample_images(sample))

        # get avg img repr of node
        nid = nid2score[0][0]
        avg_img_repr = self.vsc.get_node_embedding(
            nid, self.vs_gloss_lang, approach="avg_images"
        )

        if self.normalize:
            avg_img_repr = F.normalize(avg_img_repr, dim=-1)

        # get most similar image to avg_img
        i2i = (avg_img_repr @ i_repr.T).squeeze().cpu()

        sims = torch.softmax(i2i, dim=0).numpy()
        ranks = torch.argsort(i2i, dim=0).numpy()[::-1]

        return sims, ranks

    def _single_compute_i2i_sims_single_images(
        self, sample: VWSDSampleBase, nid2score: List[Tuple[int, float]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def _run_single(self) -> Dict[str, List[str]]:
        results = {
            "idx": [],
            "sims": [],
            "ranks": [],
        }

        for sample in tqdm(self.ds, disable=not self.show_progress_bar):
            # get best matching VisualSem node via sample text
            nid2score = self.vsc.get_best_matching_visual_sem_nodes(
                sample,
                text_or_images="text",
                vs_embeddings_approach=self.vs_embeddings_approach,
                vs_gloss_lang=self.vs_gloss_lang,
                top_k=1,
            )

            if self.i2i_single_embs:
                (
                    sims,
                    ranks,
                ) = self._single_compute_i2i_sims_single_images(sample, nid2score)
            else:
                sims, ranks = self._single_compute_i2i_sims_avg_images(
                    sample, nid2score
                )

            results["idx"].append(sample.idx)
            results["sims"].append(sims)
            results["ranks"].append(ranks)

        return results

    def _batch_compute_i2i_sims_avg_images(
        self, batch: Dict[str, torch.Tensor], nids2scores: List[Tuple[int, float]]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        # get avg img repr of node
        avg_img_reprs = []
        for nid, _ in nids2scores:
            avg_img_reprs.append(
                self.vsc.get_node_embedding(
                    nid, self.vs_gloss_lang, approach="avg_images"
                )
            )
        avg_img_reprs = torch.stack(avg_img_reprs).to(self.encoder.device)
        if self.normalize:
            avg_img_reprs = F.normalize(avg_img_reprs, dim=-1)

        i_reprs = self.compute_visual_repr_batch(batch["images"])  # B*10, H
        i2i = (avg_img_reprs @ i_reprs.T).squeeze().cpu()  # B, 10*B

        # apply individually for each sample (avg_img vs 10 images)
        sims = [
            torch.softmax(i2i[i, i * 10 : (i + 1) * 10], dim=0).numpy()
            for i in range(len(batch["idx"]))
        ]
        ranks = [
            torch.argsort(i2i[i, i * 10 : (i + 1) * 10], dim=0).numpy()[::-1]
            for i in range(len(batch["idx"]))
        ]

        return sims, ranks

    def _batch_compute_i2i_sims_single_images(
        self, batch: Dict[str, torch.Tensor], nids2scores: List[Tuple[int, float]]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        node_image_embs = []
        for nid, _ in nids2scores:
            node_image_embs.append(
                self.vsc.get_node_embedding(
                    nid, self.vs_gloss_lang, approach="single_images"
                )
            )
        node_image_embs = pad_sequence(node_image_embs, batch_first=True)  # B, L, H

        i_reprs = self.compute_visual_repr_batch(batch["images"])
        i_reprs = i_reprs.reshape(node_image_embs.shape[0], 10, -1)  # B, 10, H

        i2i = torch.bmm(i_reprs, node_image_embs.permute(0, 2, 1)).cpu()  # B, 10, L

        # accumulate scores
        i2i = torch.sum(i2i, dim=2)

        # adjust softmax temperature for un-normalized embs
        i2i /= 1.0 if self.normalize else 1000.0

        sims = torch.softmax(i2i, dim=1).tolist()  # B, 10
        ranks = torch.argsort(i2i, dim=1, descending=True).tolist()  # B, 10

        return sims, ranks

    def _run_batch(self, batch_size: int) -> Dict[str, List]:
        results = {
            "idx": [],
            "sims": [],
            "ranks": [],
        }

        dl = self._build_dataloader(batch_size)
        for batch in tqdm(dl, disable=not self.show_progress_bar):

            # get best matching VisualSem nodes via sample text
            nids2scores = self.vsc.get_best_matching_visual_sem_nodes(
                batch["text"],
                text_or_images="text",
                vs_embeddings_approach=self.vs_embeddings_approach,
                vs_gloss_lang=self.vs_gloss_lang,
                top_k=1,
            )

            if self.i2i_single_embs:
                (
                    sims,
                    ranks,
                ) = self._batch_compute_i2i_sims_single_images(batch, nids2scores)
            else:
                sims, ranks = self._batch_compute_i2i_sims_avg_images(
                    batch, nids2scores
                )

            results["idx"].extend(batch["idx"])
            results["sims"].extend(sims)
            results["ranks"].extend(ranks)

        return results


class VisualSemImageFirst(Algorithm):
    def __init__(
        self,
        encoder: ICLIPEncoder,
        ds: VWSDDataset,
        vsc: VisualSemClient,
        vs_gloss_lang: str = "en",
        t2t: bool = False,
        single_images_embs: bool = False,
        normalize: bool = True,
        show_progress_bar: bool = True,
    ):
        super().__init__(
            "VisualSemImageFirst", encoder, ds, normalize, show_progress_bar
        )
        self.vsc = vsc
        self.vs_gloss_lang = vs_gloss_lang
        self.t2t = t2t
        self.single_images_embs = single_images_embs

    def get_sample_text(self, sample: VWSDSampleBase) -> str | List[str]:
        return sample.caption

    def _run_single(self) -> Dict[str, List[str]]:
        results = {
            "idx": [],
            "sims": [],
            "ranks": [],
        }

        for sample in tqdm(self.ds, disable=not self.show_progress_bar):
            t_repr = self.compute_textual_repr(sample.caption)

            # get best matching VisualSem nodes via sample images
            nids2scores = self.vsc.get_best_matching_visual_sem_nodes(
                sample,
                text_or_images="images",
                vs_embeddings_approach="avg_images",
                vs_gloss_lang=self.vs_gloss_lang,
                top_k=1,
            )
            if self.t2t:
                # get avg gloss reprs of nodes
                avg_gloss_reprs = []
                for nid, _ in nids2scores:
                    bnid = self.vsc.vs[nid]["bnid"]
                    avg_gloss_reprs.append(
                        self.vsc.vs_embeddings[self.vs_gloss_lang]["avg_glosses"][bnid]
                    )
                avg_gloss_reprs = torch.stack(avg_gloss_reprs).to(
                    self.encoder.device
                )  # 10, H
                if self.normalize:
                    avg_gloss_reprs = F.normalize(avg_gloss_reprs, dim=-1)

                # get most similar avg_gloss to t_repr
                t2t = (avg_gloss_reprs @ t_repr.T).squeeze().cpu()  # 10
                sim = t2t
            else:
                # get avg img repr of node
                avg_img_reprs = []
                for nid, _ in nids2scores:
                    bnid = self.vsc.vs[nid]["bnid"]
                    avg_img_reprs.append(
                        self.vsc.vs_embeddings[self.vs_gloss_lang]["avg_images"][bnid]
                    )
                avg_img_reprs = torch.stack(avg_img_reprs).to(self.encoder.device)
                if self.normalize:
                    avg_img_reprs = F.normalize(avg_img_reprs, dim=-1)

                # get most similar image to avg_imgs
                i2t = (avg_img_reprs @ t_repr.T).squeeze().cpu()  # 10
                sim = i2t

            sims = torch.softmax(sim, dim=0).numpy()
            ranks = torch.argsort(sim, dim=0).numpy()[::-1]

            results["idx"].append(sample.idx)
            results["sims"].append(sims)
            results["ranks"].append(ranks)

        return results

    def _run_batch(self, batch_size: int) -> Dict[str, List]:
        logger.warning("Batching is not implemented! Running in single mode!")
        return self._run_single()
        raise NotImplementedError
        results = {
            "idx": [],
            "sims": [],
            "ranks": [],
        }

        dl = self._build_dataloader(batch_size)
        for batch in tqdm(dl, disable=not self.show_progress_bar):
            i_reprs = self.compute_visual_repr_batch(batch["images"])  # B*10, H

            # get best matching VisualSem nodes via sample text
            nids2scores = self.vsc.get_best_matching_visual_sem_nodes(
                batch["text"],
                text_or_images="text",
                vs_embeddings_approach="avg_glosses",
                vs_gloss_lang=self.vs_gloss_lang,
                top_k=1,
            )
            # get avg img repr of node
            avg_img_reprs = []
            for nid, _ in nids2scores:
                bnid = self.vsc.vs[nid]["bnid"]
                avg_img_reprs.append(
                    self.vsc.vs_embeddings[self.vs_gloss_lang]["avg_images"][bnid]
                )
            avg_img_reprs = torch.stack(avg_img_reprs).to(self.encoder.device)
            if self.normalize:
                avg_img_reprs = F.normalize(avg_img_reprs, dim=-1)

            i2i = (avg_img_reprs @ i_reprs.T).squeeze().cpu()  # B, 10*B

            # apply individually for each sample (avg_img vs 10 images)
            sims = [
                torch.softmax(i2i[i, i * 10 : (i + 1) * 10], dim=0)
                for i in range(len(batch["idx"]))
            ]
            ranks = [
                torch.argsort(i2i[i, i * 10 : (i + 1) * 10], dim=0)[::-1]
                for i in range(len(batch["idx"]))
            ]

            results["idx"].extend(batch["idx"])
            results["sims"].extend(sims)
            results["ranks"].extend(ranks)

        return results
