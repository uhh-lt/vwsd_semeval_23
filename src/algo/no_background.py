from src.algo.algo import Algorithm
from src.model.clip_encoder import ICLIPEncoder
from src.vwsd_dataset import VWSDDataset, VWSDSampleBase


class NoBackground(Algorithm):
    def __init__(
        self,
        encoder: ICLIPEncoder,
        ds: VWSDDataset,
        normalize: bool = True,
        show_progress_bar: bool = True,
    ):
        super().__init__("NoBackground", encoder, ds, normalize, show_progress_bar)

    def get_sample_text(self, sample: VWSDSampleBase) -> str:
        return sample.caption
