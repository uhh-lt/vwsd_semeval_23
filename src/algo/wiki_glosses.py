from src.algo.algo import Algorithm
from src.model.clip_encoder import ICLIPEncoder
from src.vwsd_dataset import VWSDDataset, VWSDSampleBase
from src.wiki_client import WikiClient


class WikiGlosses(Algorithm):
    def __init__(
        self,
        encoder: ICLIPEncoder,
        ds: VWSDDataset,
        wc: WikiClient,
        normalize: bool = True,
        show_progress_bar: bool = True,
    ):
        super().__init__("WikiGlosses", encoder, ds, normalize, show_progress_bar)
        self.wc = wc

    def get_sample_text(self, sample: VWSDSampleBase) -> str:
        wiki_sum = self.wc.get_wiki_summary(sample=sample)
        if wiki_sum == '':
            return sample.caption
        return wiki_sum
