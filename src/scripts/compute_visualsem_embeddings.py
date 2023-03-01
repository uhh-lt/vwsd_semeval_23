from pathlib import Path
from typing import Dict, List, Tuple
import fire
from loguru import logger
from matplotlib.pyplot import re
import torch
from torchvision.io import ImageReadMode, read_image
from tqdm.auto import tqdm

from model.clip_encoder import ICLIPEncoder, build_clip_encoder
from visualsem.visualsem_dataset_nodes import VisualSemNodesDataset


def get_glosses_and_images(
    vs: VisualSemNodesDataset,
    node_idx: int,
    gloss_lang: str = "en",
    max_images: int = 50,
) -> Tuple[str, List[str], List[torch.Tensor]]:
    node_bnid = vs.bnids[node_idx]
    node = vs[node_idx]

    # get the glosses
    if gloss_lang not in node["glosses"]:
        logger.warning(
            f"There are no glosses for language '{gloss_lang}'. Using 'en' as fallback!"
        )
        glosses = node["glosses"]["en"]
    else:
        glosses = node["glosses"][gloss_lang]

    # get the images
    images = []
    for img in vs.get_node_images_by_bnid(node_bnid)[:max_images]:
        try:
            images.append(read_image(str(img), mode=ImageReadMode.RGB))
        except Exception as e:
            logger.warning(f"Cannot read '{img}' ({e})!")
            continue

    return node_bnid, glosses, images


def compute_visualsem_node_embeddings(
    vs: VisualSemNodesDataset,
    clip_encoder: ICLIPEncoder,
    gloss_lang: str = "en",
    max_images: int = 50,
    resume: bool = True,
) -> Dict[str, Dict[str, torch.Tensor]]:
    approaches = ["single_glosses", "single_images", "avg_glosses", "avg_images"]

    out_p = {
        a: vs.root_path
        / "embeddings"
        / f"{clip_encoder.model_name.replace('/', '-')}_{a}_{gloss_lang}.pt"
        for a in approaches
    }

    data = {a: dict() for a in approaches}
    if resume:
        data = {a: load_embeddings(file=out_p[a]) for a in approaches}

    for node_idx in tqdm(range(len(vs)), desc="Computing VisualSem Node Embeddings"):
        node_bnid = vs.bnids[node_idx]
        # if all embeddings for the node have been already computed, skip!
        if all([node_bnid in data[a] for a in approaches]):
            continue

        node_bnid, glosses, images = get_glosses_and_images(
            vs, node_idx, gloss_lang, max_images
        )

        with torch.no_grad():
            # compute single gloss embs
            single_gloss_embs = clip_encoder.encode_text(glosses, normalize=False)
            # compute avg gloss embs
            avg_glosses_emb = torch.mean(single_gloss_embs, dim=0)
            # save
            data["single_glosses"][node_bnid] = single_gloss_embs.cpu()
            data["avg_glosses"][node_bnid] = avg_glosses_emb.cpu()

            # compute single image embs
            single_img_embs = clip_encoder.encode_images(images, normalize=False)
            # compute avg image embs
            avg_images_emb = torch.mean(single_img_embs, dim=0)
            # save
            data["single_images"][node_bnid] = single_img_embs.cpu()
            data["avg_images"][node_bnid] = avg_images_emb.cpu()

        if node_idx > 0 and node_idx % 1000 == 0:
            for approach, embs in data.items():
                persist_embeddings(embeddings=embs, out_p=out_p[approach])

    for approach, embs in data.items():
        persist_embeddings(embeddings=embs, out_p=out_p[approach])

    return data


def persist_embeddings(embeddings: Dict[str, torch.Tensor], out_p: Path):
    if not out_p.parent.exists():
        out_p.parent.mkdir(parents=True)

    torch.save(embeddings, out_p)
    logger.info(f"Persisted {len(embeddings)} embeddings at: {out_p}")


def load_embeddings(file: Path) -> Dict[str, torch.Tensor]:
    if not file.exists():
        logger.error(f"Cannot read embeddings at{file}!")
        return dict()

    embeddings = torch.load(file)
    logger.info(f"Loaded {len(embeddings)} embeddings from {file}")
    return embeddings


def get_visualsem_dataset(visualsem_dataset_p: Path) -> VisualSemNodesDataset:
    assert (
        visualsem_dataset_p.exists()
    ), f"Cannot find VisualSem Dataset at {visualsem_dataset_p}!"
    nodes_json = visualsem_dataset_p / "nodes.v2.json"
    glosses_json = visualsem_dataset_p / "gloss_files" / "nodes.glosses.json"
    tuples_json = visualsem_dataset_p / "tuples.v2.json"
    path_to_images = visualsem_dataset_p / "images"
    return VisualSemNodesDataset(nodes_json, glosses_json, tuples_json, path_to_images)


def main(
    visualsem_dataset_p: str | Path,
    model_name: str,
    device: str = "cuda:0",
    gloss_lang: str = "en",
    max_images: int = 50,
    resume: bool = True
):
    visualsem_dataset_p = Path(visualsem_dataset_p).resolve()
    vs = get_visualsem_dataset(visualsem_dataset_p)
    enc = build_clip_encoder(model_name=model_name, device=device)

    compute_visualsem_node_embeddings(
        vs=vs, clip_encoder=enc, gloss_lang=gloss_lang, max_images=max_images, resume=resume
    )


if __name__ == "__main__":
    fire.Fire(main)
