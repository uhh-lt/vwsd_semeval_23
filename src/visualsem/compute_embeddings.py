from typing import Dict, List, Optional, Tuple

from PIL import Image
import pyrootutils
import torch
from tqdm.auto import tqdm

root = pyrootutils.setup_root(".", pythonpath=True, cwd=False)

from clip_model_wrapper import CLIPModelWrapper
from visualsem_dataset_nodes import VisualSemNodesDataset


def get_glosses_and_images(
    vs: VisualSemNodesDataset, node_idx: int, gloss_langs: Optional[List[str]] = None
) -> Tuple[str, List[str], List[Image.Image]]:
    if gloss_langs is None or len(gloss_langs) == 0:
        gloss_langs = ["en"]

    node_bnid = vs.bnids[node_idx]
    node = vs[node_idx]

    # get the glosses
    glosses = []
    for l in gloss_langs:
        if l in node["glosses"]:
            glosses += node["glosses"][l]

    # get the images
    images = [Image.open(img) for img in vs.get_node_images_by_bnid(node_bnid)]

    return node_bnid, glosses, images


def compute_node_embeddings(
    vs: VisualSemNodesDataset,
    clip_model: CLIPModelWrapper,
    approaches: List[str],
    gloss_langs: Optional[List[str]] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    data = {a: dict() for a in approaches}
    for node_idx in tqdm(range(len(vs))):
        node_bnid, glosses, images = get_glosses_and_images(vs, node_idx, gloss_langs)

        with torch.no_grad():
            # compute avg gloss embs
            gloss_feats = clip_model.encode_text(glosses)
            avg_glosses_emb = torch.mean(gloss_feats, dim=0)
            data[approaches[0]][node_bnid] = avg_glosses_emb.cpu()

            # compute avg image embs
            img_feats = clip_model.encode_images(images)
            avg_images_emb = torch.mean(img_feats, dim=0)
            data[approaches[1]][node_bnid] = avg_images_emb.cpu()

            # compute avg glosses and images feats
            avg_glosses_and_images_emb = torch.mean(
                torch.concat((gloss_feats, img_feats)), dim=0
            )
            data[approaches[2]][node_bnid] = avg_glosses_and_images_emb.cpu()

    return data


if __name__ == "__main__":
    nodes_json = root / "dataset" / "nodes.v2.json"
    glosses_json = root / "dataset" / "gloss_files" / "nodes.glosses.json"
    tuples_json = root / "dataset" / "tuples.v2.json"
    path_to_images = root / "dataset" / "images"
    vs = VisualSemNodesDataset(nodes_json, glosses_json, tuples_json, path_to_images)

    # TODO fire for cli args!
    # clip_model = "laion/CLIP-ViT-H-14-frozen-xlm-roberta-large-laion5B-s13B-b90k"
    clip_model = "laion/CLIP-ViT-L-14-laion2B-s32B-b82K"
    # clip_model = "sentence-transformers/clip-ViT-B-32-multilingual-v1"

    gloss_langs = ["en"]  # , "de", "fr", "es", "ru", "pt"]
    device = "cuda:0"
    approaches = ["avg_glosses", "avg_images", "avg_glosses_and_images"]
    normalize = True

    print(f"Using CLIP {clip_model}")

    model = CLIPModelWrapper(model_name=clip_model, device=device)
    data = compute_node_embeddings(vs, model, approaches, gloss_langs)
    for approach in approaches:
        out = (
            root
            / "dataset"
            / f"embs{'/normalized' if normalize else ''}"
            / f'{clip_model}_{approach}{"_" + "+".join(gloss_langs) if "glosses" in approach else ""}.pt'
        )
        if not out.parent.exists():
            out.parent.mkdir(parents=True)

        torch.save(data[approach], out)
        print(f"Persisted Embeddings at: {out}")
