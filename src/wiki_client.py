from pathlib import Path
from loguru import logger

import pandas as pd
import wikipedia
from tqdm import tqdm


from src.vwsd_dataset import VWSDDataset, VWSDSampleBase


class WikiClient:
    def __init__(self,
                 summaries_df_p: Path,
                 max_summary_len: int = 1000,
                 language: str = 'en'):
        self.summaries_df_p = summaries_df_p
        self.max_summary_len = max_summary_len
        self.ckpt_step = 100

        wikipedia.set_rate_limiting(True)
        wikipedia.set_lang(language)

        self.language = language
        self._dataframe_cols = ['idx', 'word', 'context', 'summary']

        self.summaries = self._load_summaries_dataframe_checkpoint(summaries_df_p)

    def get_wiki_summary(self, sample: VWSDSampleBase) -> str:
        res = self._get_sample_row(sample=sample)
        if len(res) == 0:
            self._download_wikipedia_summary(sample)

        res = self._get_sample_row(sample=sample)
        return res.summary.values[0]

    def download_wikipedia_summaries(self, ds: VWSDDataset):
        for i, s in tqdm(enumerate(ds), total=len(ds), desc="Downloading Wikipedia Summaries"):
            self.get_wiki_summary(sample=s)
            if i % self.ckpt_step == 0:
                self._persist_summaries_dataframe_checkpoint()
        self._persist_summaries_dataframe_checkpoint()

    def _get_sample_row(self, sample: VWSDSampleBase) -> pd.DataFrame:
        return self.summaries.query(f'idx == {sample.idx} & word == "{sample.word}" & context == "{sample.context}"')

    def _download_wikipedia_summary(self, sample: VWSDSampleBase):
        summary = ''
        try:
            try:
                # try direct page resolve via context
                page = wikipedia.page(title=sample.context, auto_suggest=True, redirect=True, preload=False)
                summary = page.summary[:self.max_summary_len]
            except Exception as e:
                if isinstance(e, (wikipedia.PageError, wikipedia.DisambiguationError)):
                    # try to find the page via search using the context
                    search_results = wikipedia.search(sample.context, suggestion=False)
                    if len(search_results) == 0:
                        # try to find the page via search using the word
                        search_results = wikipedia.search(sample.word, suggestion=False)
                        if len(search_results) == 0:
                            logger.warning(f"Cannot find any Wikipedia Entry for: {sample.context}")
                            raise e

                    # pick the first search result
                    sr = search_results[0]

                    try:
                        # get the summary of the search result
                        summary = wikipedia.summary(sr, auto_suggest=False)
                    except Exception as e:
                        if isinstance(e, wikipedia.DisambiguationError):
                            try:
                                # add the context to the search result to avoid disambiguation error
                                summary = wikipedia.summary(f"{sr} {sample.context}", auto_suggest=True)
                            except Exception as e:
                                if isinstance(e, wikipedia.PageError):
                                    # try again without auto suggest
                                    summary = wikipedia.summary(f"{sr} {sample.context}", auto_suggest=False)
                                else:
                                    raise e
                        else:
                            raise e
                else:
                    raise e
        except Exception as e:
            logger.error(f"Cannot retrieve Wikipedia summary for: {sample.context}. Error: {e}")
            summary = ''
        finally:
            row = self._get_sample_row(sample)
            if len(row) == 0:
                old_idx = self.summaries.index.max()
                new_index = 0 if pd.isnull(old_idx) else old_idx + 1
            else:
                new_index = row.index[0]
            self.summaries.loc[new_index] = (sample.idx, sample.word, sample.context, summary)

    def _persist_summaries_dataframe_checkpoint(self):
        logger.info(
            f"Persisting WikiSummaries DataFrame checkpoint with {len(self.summaries)} summaries at {self.summaries_df_p}"
        )

        if not self.summaries_df_p.parent.exists():
            self.summaries_df_p.parent.mkdir(parents=True)
        self.summaries.to_parquet(self.summaries_df_p)

    def _load_summaries_dataframe_checkpoint(self, path: Path | None):
        empty = pd.DataFrame(columns=self._dataframe_cols)
        if path is None or not path.exists():
            logger.warning(f"Cannot read WikiSummaries DataFrame at {path}!")
            return empty
        df = pd.read_parquet(str(path))
        if not all(c in df.columns for c in self._dataframe_cols):
            logger.warning(f"WikiSummaries DataFrame at {path} does not contain all mandatory columns!")
            return empty
        logger.info(f"Loaded WikiSummaries DataFrame with {len(df)} summaries!")
        return df

