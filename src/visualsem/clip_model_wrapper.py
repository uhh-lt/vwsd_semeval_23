from enum import Enum
from typing import List, Optional, Tuple, Union

from PIL import Image
import open_clip
from open_clip.tokenizer import HFTokenizer
from sentence_transformers import SentenceTransformer
import torch
import torchvision.transforms as T
from transformers import CLIPModel, CLIPProcessor
import torch.nn.functional as F


class ClipModelType(str, Enum):
    SENTENCE_TRANSFORMERS = "sentence-transformers"
    OPEN_CLIP = "open-clip"
    TRANSFORMERS = "transformers"
    CLIP = "clip"


class CLIPModelWrapper:
    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.device = device

        self._model_type: Optional[ClipModelType] = None
        self.model = self._load()

    def _set_model_type(self, mt: ClipModelType):
        print(f"Setting CLIP Model Type to: {mt}")
        self._model_type = mt


    def _load(
        self,
    ) -> Union[
        Tuple[SentenceTransformer, SentenceTransformer],
        Tuple[T.Compose, HFTokenizer, open_clip.CLIP],
        Tuple[CLIPProcessor, CLIPModel],
    ]:
        print(f"Loading {self.model_name} to device {self.device}")
        if "sentence-transformers" in self.model_name:
            self._set_model_type(ClipModelType.SENTENCE_TRANSFORMERS)
            return (
                SentenceTransformer(self.model_name, device=self.device),
                SentenceTransformer("clip-ViT-B-32", device=self.device),
            )
        elif "laion" in self.model_name or "openai" in self.model_name:
            if "xlm-roberta-large" in self.model_name:
                self._set_model_type(ClipModelType.OPEN_CLIP)
                model, _, preprocess = open_clip.create_model_and_transforms(
                    "xlm-roberta-large-ViT-H-14",
                    "frozen_laion5b_s13b_b90k",
                    device=self.device,
                )
                tokenizer = open_clip.get_tokenizer("xlm-roberta-large-ViT-H-14")

                return preprocess, tokenizer, model
            elif "xlm-roberta-base" in self.model_name:
                self._set_model_type(ClipModelType.OPEN_CLIP)
                model, _, preprocess = open_clip.create_model_and_transforms(
                    "xlm-roberta-base-ViT-B-32", "laion5b_s13b_b90k", device=self.device
                )
                tokenizer = open_clip.get_tokenizer("xlm-roberta-base-ViT-B-32")
                return preprocess, tokenizer, model
            else:
                self._set_model_type(ClipModelType.TRANSFORMERS)
                return (
                    CLIPProcessor.from_pretrained(self.model_name),
                    CLIPModel.from_pretrained(self.model_name).to(self.device),
                )
        else:
            raise NotImplementedError

    def _apply_image_transform_for_model_type(
        self, imgs: List[Image.Image | torch.Tensor]
    ) -> List[Image.Image | torch.Tensor]:
        to_tensor = T.ToTensor()
        to_pil_image = T.ToPILImage()
        if self._model_type == ClipModelType.TRANSFORMERS:
            return [to_tensor(img) if isinstance(img, Image.Image) else img for img in imgs]
        elif self._model_type == ClipModelType.SENTENCE_TRANSFORMERS or self._model_type == ClipModelType.OPEN_CLIP:
            return [to_pil_image(img) if isinstance(img, torch.Tensor) else img for img in imgs]
        elif self._model_type == ClipModelType.CLIP:
            raise NotImplementedError
        else:
            raise NotImplementedError


    def encode_images(
        self, imgs: List[Union[torch.Tensor, Image.Image]], normalize: bool = False
    ) -> torch.Tensor:
        imgs = self._apply_image_transform_for_model_type(imgs)
        with torch.no_grad():
            if self._model_type == ClipModelType.TRANSFORMERS:
                processor: CLIPProcessor = self.model[0]
                model: CLIPModel = self.model[1]

                img_prep = processor.feature_extractor(imgs, return_tensors="pt")
                img_prep = img_prep.to(self.device)
                img_feats = model.get_image_features(
                    pixel_values=img_prep["pixel_values"],
                    output_attentions=False,
                    output_hidden_states=False,
                )  # type: ignore

                if normalize:
                    img_feats = F.normalize(img_feats, dim=-1)
            elif self._model_type == ClipModelType.OPEN_CLIP:
                processor: T.Compose = self.model[0]  # type: ignore
                tokenizer: HFTokenizer = self.model[1]  # type: ignore
                model: open_clip.CLIP = self.model[2]  # type: ignore

                img_prep = torch.stack([processor(img) for img in imgs]).to(self.device)
                img_feats = model.encode_image(img_prep, normalize=normalize)

            elif self._model_type == ClipModelType.SENTENCE_TRANSFORMERS:
                img_feats = self.model[1].encode(
                    sentences=imgs,
                    batch_size=64,
                    show_progress_bar=False,
                    normalize_embeddings=normalize,
                    convert_to_tensor=True,
                    device=self.device,
                )  # type: ignore
            else:
                raise NotImplementedError

        return img_feats

    def encode_text(self, text: List[str], normalize: bool = False) -> torch.Tensor:
        with torch.no_grad():
            if self._model_type == ClipModelType.TRANSFORMERS:
                processor: CLIPProcessor = self.model[0]
                model: CLIPModel = self.model[1]

                txt_prep = processor.tokenizer.batch_encode_plus(
                    text,
                    add_special_tokens=True,
                    padding=True,
                    truncation=True,
                    max_length=77,
                    return_tensors="pt",
                ).to(self.device)
                txt_feats = model.get_text_features(
                    input_ids=txt_prep["input_ids"],
                    attention_mask=txt_prep["attention_mask"],
                    output_attentions=False,
                    output_hidden_states=False,
                )  # type: ignore
                if normalize:
                    txt_feats = F.normalize(txt_feats, dim=-1)

            elif self._model_type == ClipModelType.OPEN_CLIP:
                processor: T.Compose = self.model[0]
                tokenizer: HFTokenizer = self.model[1]
                model: open_clip.CLIP = self.model[2]

                txt_prep = tokenizer(text).to(self.device)
                txt_feats = model.encode_text(txt_prep, normalize=normalize)

            elif self._model_type == ClipModelType.SENTENCE_TRANSFORMERS:
                txt_feats = self.model[0].encode(
                    sentences=text,
                    batch_size=64,
                    show_progress_bar=False,
                    normalize_embeddings=normalize,
                    convert_to_tensor=True,
                    device=self.device,
                )
            else:
                raise NotImplementedError

        return txt_feats
