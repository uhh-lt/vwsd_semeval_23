from abc import ABC, abstractmethod
from datetime import datetime
import itertools as it
import json
from pathlib import Path
from typing import Dict, List
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.metrics import top_k_accuracy_score
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model.clip_encoder import ICLIPEncoder
from src.vwsd_dataset import VWSDDataset, VWSDSampleBase


class Algorithm(ABC):
    @abstractmethod
    def __init__(
        self,
        name: str,
        encoder: ICLIPEncoder,
        ds: VWSDDataset,
        normalize: bool = True,
        show_progress_bar: bool = True,
    ):
        self.name = name
        self.encoder = encoder
        self.ds = ds
        self.show_progress_bar = show_progress_bar
        self.normalize = normalize
        logger.info(
            f"Instantiating VWSD Algorithm '{self.name}' with {self.encoder.model_name}!"
        )

    def compute_textual_repr(self, txt: str) -> torch.Tensor:
        return self.encoder.encode_text([txt], normalize=self.normalize)

    def compute_visual_repr(self, images: List[torch.Tensor]) -> torch.Tensor:
        return self.encoder.encode_images(images, normalize=self.normalize)

    def compute_textual_repr_batch(self, txt_batch: List[str]) -> torch.Tensor:
        return self.encoder.encode_text(txt_batch, normalize=self.normalize)

    def compute_visual_repr_batch(
        self, images_batch: List[List[torch.Tensor]]
    ) -> torch.Tensor:
        imgs = list(it.chain.from_iterable([img for img in images_batch]))
        reprs = self.encoder.encode_images(imgs, normalize=self.normalize)
        return reprs

    @abstractmethod
    def get_sample_text(self, sample: VWSDSampleBase) -> str | List[str]:
        raise NotImplementedError

    def get_sample_images(self, sample: VWSDSampleBase) -> List[torch.Tensor]:
        return sample.images

    def __batch_collate_fn(
        self,
        batch: List[VWSDSampleBase],
    ) -> Dict[str, List[int | str | torch.Tensor]]:
        data = {"idx": [], "text": [], "num_texts": [], "images": []}
        for s in batch:
            data["idx"].append(s.idx)
            data["images"].append(self.get_sample_images(s))

            st = self.get_sample_text(s)
            if isinstance(st, str):
                data["text"].append(st)
                data["num_texts"].append(1)
            elif isinstance(st, List):
                data["text"].extend(st)
                data["num_texts"].append(len(st))
            else:
                raise ValueError(
                    f"Sample Text is neither of type str or List but: {type(st)}"
                )

        return data

    def _build_dataloader(self, batch_size: int) -> DataLoader:
        return DataLoader(
            self.ds,
            batch_size,
            collate_fn=self.__batch_collate_fn,
            shuffle=False,
            num_workers=16,
            pin_memory=True,
        )

    def _run_batch(self, batch_size: int) -> Dict[str, List]:
        results = {
            "idx": [],
            "sims": [],
            "ranks": [],
        }

        dl = self._build_dataloader(batch_size)
        for batch in tqdm(dl, disable=not self.show_progress_bar):
            t_reprs = self.compute_textual_repr_batch(
                batch["text"]
            )  # sum(B[num_texts]), H
            i_reprs = self.compute_visual_repr_batch(batch["images"])  # B*10, H

            if sum(batch["num_texts"]) != len(batch):
                # compute the average of the textual reprs for the samples
                reprs = []
                for i in range(len(batch["num_texts"])):
                    start = 0 if i == 0 else sum(batch["num_texts"][:i])
                    stop = sum(batch["num_texts"][: i + 1])
                    to_mean = t_reprs[start:stop, :]
                    reprs.append(torch.mean(to_mean, dim=0))
                t_reprs = torch.stack(reprs)  # B, H

            t2i = (t_reprs @ i_reprs.T).squeeze().cpu()  # B, 10*B

            # apply individually for each sample (text vs 10 images)
            sims = [
                torch.softmax(t2i[i, i * 10:(i + 1) * 10], dim=0).numpy()
                for i in range(len(batch["idx"]))
            ]
            ranks = [
                torch.argsort(t2i[i, i * 10:(i + 1) * 10], dim=0).numpy()[::-1]
                for i in range(len(batch["idx"]))
            ]

            results["idx"].extend(batch["idx"])
            results["sims"].extend(sims)
            results["ranks"].extend(ranks)

        return results

    def _run_single(self) -> Dict[str, List[str]]:
        results = {
            "idx": [],
            "sims": [],
            "ranks": [],
        }

        for sample in tqdm(self.ds, disable=not self.show_progress_bar):
            st = self.get_sample_text(sample)
            if isinstance(st, str):
                t_repr = self.compute_textual_repr(st)
            elif isinstance(st, List):
                # compute the average of the textual reprs for the samples
                reprs = []
                for s in st:
                    reprs.append(self.compute_textual_repr(s))
                t_repr = torch.mean(torch.stack(reprs), dim=0)
            else:
                raise ValueError(
                    f"Sample Text is neither of type str or List but: {type(st)}"
                )

            i_repr = self.compute_visual_repr(self.get_sample_images(sample))

            t2i = (t_repr @ i_repr.T).squeeze().cpu()

            sims = torch.softmax(t2i, dim=0).numpy()
            ranks = torch.argsort(t2i, dim=0).numpy()[::-1]

            results["idx"].append(sample.idx)
            results["sims"].append(sims)
            results["ranks"].append(ranks)

        return results

    def run(self, batch_size: int = 1, eval: bool = True, persist_dir_p: Path | None = None) -> pd.DataFrame:
        with torch.no_grad():
            if batch_size > 1:
                results = self._run_batch(batch_size)
            else:
                results = self._run_single()

        df = pd.DataFrame.from_dict(results)

        if self.ds.has_gold:
            gold = self.ds.get_gold()
            gold = np.array([gold[idx] for idx in df.idx])
            df['gold'] = gold

        df['algo'] = self.name
        df.rename(columns={"idx": "vwsd_sample_idx"})

        if eval:
            if not self.ds.has_gold:
                logger.error(f"Cannot evaluate {self.name} because the provided VWSDDataset has no gold labels!")
            else:
                self.eval(df)

        if persist_dir_p is not None:
            self.__persist(persist_dir_p, df)

        return df

    def eval(self, results: pd.DataFrame, persist_dir_p: Path | None = None):
        assert all(
            c in results.columns for c in ["idx", "sims", "ranks"]
        ), "Cannot evaluate because not all mandatory columns are present!"
        assert (
            self.ds.has_gold
        ), "Cannot evaluate because the Dataset doesn't contain gold images!"

        logger.info("Computing Hits@K scores...")
        gold = results.gold.to_numpy()
        sims = np.stack(results.sims.to_numpy())
        scores = {}
        for k in [1, 3, 5, 10]:
            scores[k] = top_k_accuracy_score(gold, sims, k=k, labels=list(range(10)))
        logger.info(json.dumps(scores, indent=2))

        if persist_dir_p is not None:
            self.__persist(persist_dir_p, scores)

    def __persist(self, persist_dir_p: Path, results_or_scores: pd.DataFrame | Dict[str, str]) -> Path:
        persist_dir_p = Path(persist_dir_p)
        if not persist_dir_p.parent.exists():
            persist_dir_p.parent.mkdir(parents=True)
        now = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        out_fn = persist_dir_p / f"{self.name}_{now}_{'scores.json' if isinstance(results_or_scores, Dict) else 'results.df.pq'}"

        logger.info(f"Persisting at {out_fn}")

        if isinstance(results_or_scores, Dict):
            with open(out_fn, "w") as f:
                json.dump(results_or_scores, f, indent=2)
        elif isinstance(results_or_scores, pd.DataFrame):
            results_or_scores.to_parquet(out_fn)

        return out_fn
