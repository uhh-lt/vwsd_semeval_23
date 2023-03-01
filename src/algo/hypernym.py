from pathlib import Path
import pandas as pd

from src.algo.algo import Algorithm
from src.model.clip_encoder import ICLIPEncoder
from src.vwsd_dataset import VWSDDataset, VWSDSampleBase


class Hypernym(Algorithm):
    def __init__(
        self,
        encoder: ICLIPEncoder,
        ds: VWSDDataset,
        hypernym_df_p: Path,
        normalize: bool = True,
        show_progress_bar: bool = True,
    ):
        super().__init__("Hypernym", encoder, ds, normalize, show_progress_bar)

        assert hypernym_df_p.exists(), f"Cannot read Hypernym DataFrame from {hypernym_df_p}"
        self.hypernyms = pd.read_parquet(hypernym_df_p)

    def get_sample_text(self, sample: VWSDSampleBase) -> str:
        ctx = sample.context.replace(sample.word, "").strip()

        hypers = set()
        for hyper in self.hypernyms.loc[sample.idx].t5_preds:
            hypers.add(hyper.strip())

        return f'An image of a "{sample.word}" in the context of "{ctx}". Possible hypernyms: {", ".join(hypers)}.'
