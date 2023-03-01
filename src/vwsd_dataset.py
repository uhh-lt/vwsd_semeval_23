from pathlib import Path
from typing import Optional, Callable, List, Dict

import pandas as pd
import numpy as np
import torch
from torchvision.datasets import VisionDataset
from torchvision.io import ImageReadMode, read_image
from loguru import logger
from googletrans import Translator


class VWSDSampleBase:
    def __init__(
        self,
        idx: int,
        word: str,
        context: str,
        images: List[torch.Tensor],
        caption_template: str,
        replace_word_in_context: bool = False,
    ):
        self.idx = idx
        self.word = word
        self.context = context

        assert len(images) == 10
        self.images = images

        self.replace_word_in_context = replace_word_in_context
        self.caption_template = caption_template

    @property
    def caption(self) -> str:
        ctx = self.context
        if self.replace_word_in_context:
            ctx = self.context.replace(self.word, "").strip()

        text = self.caption_template.replace("<WORD>", self.word).replace(
            "<CONTEXT>", ctx
        )
        text = text.replace(u'\u200e', '')  # remove ltr sign from farsi text
        return text

    def __repr__(self):
        return f"VWSDSampleBase(word='{self.word}', context='{self.context}')"

    def __str__(self):
        return self.__repr__()


class VWSDGoldSample(VWSDSampleBase):
    def __init__(
        self,
        idx: int,
        word: str,
        context: str,
        images: List[torch.Tensor],
        gold_idx: int,
        caption_template: str,
        replace_word_in_context: bool = False,
    ):
        super().__init__(
            idx, word, context, images, caption_template, replace_word_in_context
        )

        assert 10 > gold_idx >= 0
        self.gold_idx = gold_idx


class VWSDTestSample(VWSDSampleBase):
    def __init__(
        self,
        idx: int,
        word: str,
        context: str,
        images: List[torch.Tensor],
        caption_template: str,
        replace_word_in_context: bool = False,
    ):
        super().__init__(
            idx, word, context, images, caption_template, replace_word_in_context
        )


class VWSDDataset(VisionDataset):
    def __init__(
        self,
        dataframe_p: Path,
        images_root: Path,
        transform: Optional[Callable] = None,
        replace_word_in_context: bool = False,
        text_template: str = 'An image of a "<WORD>" as in the context "<CONTEXT>" .',
        language: str = "en",
        translate_to_en: bool = True,
    ):
        super().__init__(
            root=str(dataframe_p.parent),
            transform=transform,
            target_transform=None,
            transforms=None,
        )

        assert dataframe_p.exists(), f"Cannot read DataFrame at {dataframe_p}"
        self.dataframe = pd.read_parquet(dataframe_p)
        self.dataframe_p = dataframe_p

        assert images_root.exists(), f"Cannot read Images at {dataframe_p}"
        self.images_root = images_root

        self.has_gold = "gold" in self.dataframe.columns

        self.replace_word_in_context = replace_word_in_context
        self.text_template = text_template

        self.language = language
        self.translate_to_en = translate_to_en and not self.language == "en"
        self.translator = None
        self.translation_df = None
        self.translated_samples = 0
        if self.translate_to_en:
            self.translator = Translator()
            self.translation_df = self._create_or_load_translation_df()

        self.__item_cache: Dict[int, VWSDGoldSample | VWSDTestSample] = dict()

    def _get_translated_df_path(self) -> Path:
        return self.dataframe_p.parent / f"{self.dataframe_p.name}_en"

    def _create_or_load_translation_df(self) -> pd.DataFrame:
        tdf_p = self._get_translated_df_path()
        if tdf_p.exists():
            logger.info(f"Loading English Translation DataFrame from: {tdf_p}")
            return pd.read_parquet(tdf_p)

        tdf = self.dataframe[["word", "context"]].copy()
        tdf["word_en"] = None
        tdf["context_en"] = None
        self._persist_translation_df(tdf)

        return tdf

    def _persist_translation_df(self, tdf: pd.DataFrame) -> None:
        if tdf is None:
            return
        tdf_p = self._get_translated_df_path()
        if tdf_p.exists():
            logger.warning(f"Overwriting English Translation DataFrame at: {tdf_p}")
        tdf.to_parquet(tdf_p)
        self.translated_samples = tdf.word_en.count()
        logger.info(
            f"Persisted English Translation DataFrame with {self.translated_samples} Translated Samples at: {tdf_p}"
        )

    def translate_sample_to_en(self, sample: VWSDSampleBase) -> VWSDSampleBase:
        if self.language == "en":
            return sample
        elif self.translation_df is not None:
            word_en = self.translation_df.loc[sample.idx].word_en
            ctx_en = self.translation_df.loc[sample.idx].context_en
            if word_en is not None and ctx_en is not None:
                sample.word = word_en
                sample.context = ctx_en
                return sample

        assert (
            self.translator is not None
        ), "Cannot translate! Translator not initialized!"
        try:
            word_en = self.translator.translate(
                sample.word, src=self.language, dest="en"
            ).text
            context_en = self.translator.translate(
                sample.context, src=self.language, dest="en"
            ).text

            self.translation_df.loc[sample.idx].word_en = word_en
            self.translation_df.loc[sample.idx].context_en = context_en

            sample.word = word_en
            sample.context = context_en

            self.translated_samples += 1

            if self.translated_samples % (
                len(self.dataframe) // 10
            ) == 0 or self.translated_samples == len(self.dataframe):
                self._persist_translation_df(self.translation_df)
        except Exception as e:
            logger.error(f"Cannot translate {sample}. Error: {e}")
        finally:
            return sample

    def get_gold(self) -> Dict[int, int]:
        gold = dict()
        if not self.has_gold:
            logger.error(
                "Cant get gold from this DataSet since it contains only TestSamples and no GoldSamples"
            )
            return gold

        for _, row in self.dataframe.iterrows():
            image_names = [row[idx] for idx in row.index if "img_" in idx]
            gold_idx = image_names.index(row.gold)
            gold[row.name] = gold_idx
        return gold

    def _get_sample(self, idx: int) -> VWSDGoldSample | VWSDTestSample:
        if idx in self.__item_cache:
            return self.__item_cache[idx]

        row = self.dataframe.iloc[idx]
        image_names = [row[idx] for idx in row.index if "img_" in idx]
        images = [self.load_image(img_name=img_name) for img_name in image_names]

        if self.transform is not None:
            images = [self.transform(img) for img in images]

        if self.has_gold:
            gold_idx = image_names.index(row.gold)

            sample = VWSDGoldSample(
                idx=row.name,
                word=row.word,
                context=row.context,
                gold_idx=gold_idx,
                images=images,
                caption_template=self.text_template,
                replace_word_in_context=self.replace_word_in_context,
            )
        else:
            sample = VWSDTestSample(
                idx=row.name,
                word=row.word,
                context=row.context,
                images=images,
                caption_template=self.text_template,
                replace_word_in_context=self.replace_word_in_context,
            )

        self.__item_cache[idx] = sample
        if self.translate_to_en:
            return self.translate_sample_to_en(sample)
        return sample

    def __getitem__(self, idx: int) -> VWSDGoldSample | VWSDTestSample:
        return self._get_sample(idx)

    def __len__(self) -> int:
        return len(self.dataframe)

    def load_image(self, img_name: str) -> torch.Tensor:
        img_p = self.images_root / img_name
        assert img_p.exists(), f"Image {img_p} does not exist!"
        try:
            return read_image(str(img_p), mode=ImageReadMode.RGB)
        except Exception as e:
            logger.error(f"Cannot read image {img_p} : {e}")
            raise e
