from abc import ABC, abstractmethod
from loguru import logger

import torch
from typing import List
from PIL import Image
from sentence_transformers import SentenceTransformer
import open_clip
import clip
from transformers import CLIPModel, CLIPProcessor, AutoTokenizer
from multilingual_clip import pt_multilingual_clip
import torch.nn.functional as F
import torchvision.transforms as T


class ICLIPEncoder(ABC):
    @abstractmethod
    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.device = device
        logger.info(f"Instantiating CLIP Encoder '{model_name}'")

    @abstractmethod
    def encode_text(self, text: List[str], normalize: bool = False) -> torch.Tensor:
        pass

    @abstractmethod
    def encode_images(self, imgs: List[torch.Tensor | Image.Image], normalize: bool = False) -> torch.Tensor:
        pass


class SentenceTransformerClip(ICLIPEncoder):
    def __init__(self, model_name: str, device: str):
        super().__init__(model_name, device)

        self.text_model = SentenceTransformer(self.model_name, device=self.device)
        self.image_model = SentenceTransformer("clip-ViT-B-32", device=self.device)
        self.to_pil_image = T.ToPILImage()

    def encode_text(self, text: List[str], normalize: bool = False) -> torch.Tensor:
        with torch.no_grad():
            txt_feats = self.text_model.encode(
                sentences=text,
                batch_size=256,
                show_progress_bar=False,
                normalize_embeddings=normalize,
                convert_to_tensor=True,
                device=self.device,
            )
            return txt_feats  # type: ignore

    def encode_images(self, imgs: List[torch.Tensor | Image.Image], normalize: bool = False) -> torch.Tensor:
        with torch.no_grad():
            imgs = [self.to_pil_image(img) if isinstance(img, torch.Tensor) else img for img in imgs]
            img_feats = self.image_model.encode(
                sentences=imgs,  # type: ignore
                batch_size=256,
                show_progress_bar=False,
                normalize_embeddings=normalize,
                convert_to_tensor=True,
                device=self.device,
            )
            return img_feats  # type: ignore


class OpenClip(ICLIPEncoder):
    def __init__(self, model_name: str, device: str):
        super().__init__(model_name, device)

        if "xlm-roberta-large" in model_name:
            model, _, preprocess = open_clip.create_model_and_transforms(
                "xlm-roberta-large-ViT-H-14",
                "frozen_laion5b_s13b_b90k",
                device=self.device,
            )
            tokenizer = open_clip.get_tokenizer("xlm-roberta-large-ViT-H-14")
        elif "xlm-roberta-base" in model_name:
            model, _, preprocess = open_clip.create_model_and_transforms(
                "xlm-roberta-base-ViT-B-32", "laion5b_s13b_b90k", device=self.device
            )
            tokenizer = open_clip.get_tokenizer("xlm-roberta-base-ViT-B-32")
        else:
            raise NotImplementedError(f"Model name {self.model_name} not supported!")

        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.to_pil_image = T.ToPILImage()

    def encode_text(self, text: List[str], normalize: bool = False) -> torch.Tensor:
        with torch.no_grad():
            txt_prep = self.tokenizer(text).to(self.device)
            txt_feats = self.model.encode_text(txt_prep, normalize=normalize)  # type: ignore
            return txt_feats

    def encode_images(self, imgs: List[torch.Tensor | Image.Image], normalize: bool = False) -> torch.Tensor:
        with torch.no_grad():
            imgs = [self.to_pil_image(img) if isinstance(img, torch.Tensor) else img for img in imgs]
            img_prep = torch.stack([self.preprocess(img) for img in imgs]).to(self.device)  # type: ignore
            img_feats = self.model.encode_image(img_prep, normalize=normalize)  # type: ignore
            return img_feats


class MClip(ICLIPEncoder):
    def __init__(self, model_name: str, device: str):
        super().__init__(model_name, device)
        self.to_pil_image = T.ToPILImage()

        self.text_model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name, device=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if model_name == "M-CLIP/XLM-Roberta-Large-Vit-B-16Plus":
            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-16-plus-240", pretrained="laion400m_e32"
            )
            model.to(self.device)  # type: ignore
        elif model_name == "M-CLIP/XLM-Roberta-Large-Vit-B-32":
            model, preprocess = clip.load("ViT-B/32", device=device)
        elif model_name == "M-CLIP/XLM-Roberta-Large-Vit-L-14":
            model, preprocess = clip.load("ViT-L/14", device=device)
        else:
            raise NotImplementedError(f"Model name {self.model_name} not supported!")

        self.image_model = model
        self.preprocess = preprocess

    def encode_text(self, text: List[str], normalize: bool = False) -> torch.Tensor:
        with torch.no_grad():
            txt_feats = self.text_model.forward(text, self.tokenizer).to(self.device)  # type: ignore
            if normalize:
                txt_feats = F.normalize(txt_feats, dim=-1)
            return txt_feats

    def encode_images(self, imgs: List[torch.Tensor | Image.Image], normalize: bool = False) -> torch.Tensor:
        with torch.no_grad():
            imgs = [self.to_pil_image(img) if isinstance(img, torch.Tensor) else img for img in imgs]

            imgs = torch.stack([self.preprocess(img) for img in imgs]).to(self.device)  # type: ignore

            img_feats = self.image_model.encode_image(imgs)  # type: ignore
            if normalize:
                img_feats = F.normalize(img_feats, dim=-1)

            return img_feats


class TransformersClip(ICLIPEncoder):
    def __init__(self, model_name: str, device: str):
        super().__init__(model_name, device)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)  # type: ignore

    def encode_text(self, text: List[str], normalize: bool = False) -> torch.Tensor:
        with torch.no_grad():
            txt_prep = self.processor.tokenizer.batch_encode_plus(  # type: ignore
                text,
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(self.device)

            txt_feats = self.model.get_text_features(  # type: ignore
                input_ids=txt_prep["input_ids"],
                attention_mask=txt_prep["attention_mask"],
                output_attentions=False,
                output_hidden_states=False,
            )

            if normalize:
                txt_feats = F.normalize(txt_feats, dim=-1)

            return txt_feats

    def encode_images(self, imgs: List[torch.Tensor | Image.Image], normalize: bool = False) -> torch.Tensor:
        with torch.no_grad():
            img_prep = self.processor.feature_extractor(imgs, return_tensors="pt").to(self.device)  # type: ignore

            img_feats = self.model.get_image_features(  # type: ignore
                pixel_values=img_prep["pixel_values"],
                output_attentions=False,
                output_hidden_states=False,
            )

            if normalize:
                img_feats = F.normalize(img_feats, dim=-1)

            return img_feats


def build_clip_encoder(model_name: str, device: str = "cuda:0") -> ICLIPEncoder:
    if "sentence-transformers" in model_name:
        return SentenceTransformerClip(model_name=model_name, device=device)
    elif "M-CLIP" in model_name:
        return MClip(model_name=model_name, device=device)
    elif "laion" in model_name or "openai" in model_name:
        if "xlm-roberta" in model_name:
            return OpenClip(model_name=model_name, device=device)
        else:
            return TransformersClip(model_name=model_name, device=device)
    else:
        raise NotImplementedError(f"Model name {model_name} not supported!")
