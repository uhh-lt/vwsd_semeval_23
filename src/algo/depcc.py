from src.depcc_client import DepCCClient
from src.algo.algo import Algorithm
from src.model.clip_encoder import ICLIPEncoder
from src.vwsd_dataset import VWSDDataset, VWSDSampleBase
from typing import List


class DepCC(Algorithm):
    def __init__(
        self,
        encoder: ICLIPEncoder,
        ds: VWSDDataset,
        dc: DepCCClient,
        max_sents: int = 5,
        normalize: bool = True,
        show_progress_bar: bool = True,
    ):
        super().__init__("DepCC", encoder, ds, normalize, show_progress_bar)
        self.dc = dc
        self.max_sents = max_sents

    def get_sample_text(self, sample: VWSDSampleBase) -> str | List[str]:
        context_sents = self.dc.get_best_matching_longest_contexts(
            sample=sample, max_sents=self.max_sents
        )
        if len(context_sents) == 0:
            return sample.caption
        return context_sents
