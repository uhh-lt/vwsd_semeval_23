from datetime import datetime
import inspect
from pathlib import Path
from typing import Dict, List

import fire
from loguru import logger
import pandas as pd
import pyrootutils
import wandb

root = pyrootutils.setup_root(".", pythonpath=True)
from src.algo.algo import Algorithm
from src.algo.depcc import DepCC
from src.algo.hypernym import Hypernym
from src.algo.no_background import NoBackground
from src.algo.visualsem import VisualSemImageFirst, VisualSemTextFirst
from src.algo.wiki_glosses import WikiGlosses
from src.depcc_client import DepCCClient
from src.eval import evaluate_algorithm_results
from src.model.clip_encoder import build_clip_encoder
from src.visual_sem_client import VisualSemClient
from src.visualsem.visualsem_dataset_nodes import VisualSemNodesDataset
from src.vwsd_dataset import VWSDDataset
from src.wiki_client import WikiClient


def get_visualsem_dataset(visualsem_dataset_p: Path) -> VisualSemNodesDataset:
    visualsem_dataset_p = Path(visualsem_dataset_p).resolve()
    assert (
        visualsem_dataset_p.exists()
    ), f"Cannot find VisualSem Dataset at {visualsem_dataset_p}!"
    nodes_json = visualsem_dataset_p / "nodes.v2.json"
    glosses_json = visualsem_dataset_p / "gloss_files" / "nodes.glosses.json"
    tuples_json = visualsem_dataset_p / "tuples.v2.json"
    path_to_images = visualsem_dataset_p / "images"
    return VisualSemNodesDataset(nodes_json, glosses_json, tuples_json, path_to_images)


def main(
    clip_model_name: str = "sentence-transformers/clip-ViT-B-32-multilingual-v1",
    vwsd_dataset_p: str | Path = "data/train_phase/test_split.df.pq",
    vwsd_images_p: str | Path = "data/images/resized",
    hypernym_df_p: str | Path = "data/train_phase/test_split_t5.df.pq",
    wiki_summaries_p: str | Path = "data/train_phase/test_split_wiki_summaries.df.pq",
    visualsem_dataset_p: str | Path = "data/visualsem_dataset",
    visualsem_embeddings_p: str | Path = "data/visualsem_dataset/embeddings",
    out_p: str
    | Path = "logs/run_algorithms/",
    normalize_embeddings: bool = True,
    device: str = "cuda:0",
    gloss_lang: str = "en",
    translate_vwsd: bool = True,
    vwsd_language: str = "en",
    run_algo: List[str] = ["nobg", "wiki", "depcc", "hyper", "vsif", "vstf"],
    wandb_logging: bool = False,
) -> Path:

    vwsd_dataset_p = Path(vwsd_dataset_p)
    vwsd_images_p = Path(vwsd_images_p)
    hypernym_df_p = Path(hypernym_df_p)
    wiki_summaries_p = Path(wiki_summaries_p)
    visualsem_dataset_p = Path(visualsem_dataset_p)
    visualsem_embeddings_p = Path(visualsem_embeddings_p)
    out_p = Path(out_p) / f"{datetime.now().strftime('%Y_%m_%d-%I_%M_%S_%p')}"

    paths = {
        "vwsd_dataset_p": vwsd_dataset_p,
        "vwsd_images_p": vwsd_images_p,
        "wiki_summaries_p": wiki_summaries_p,
        "hypernym_df_p": hypernym_df_p,
        "visualsem_dataset_p": visualsem_dataset_p,
        "visualsem_embeddings_p": visualsem_embeddings_p,
        "out_p": out_p,
    }

    if wandb_logging:
        frame = inspect.currentframe()
        keys, _, _, values = inspect.getargvalues(frame)
        kwargs = {}
        for key in keys:
            if key != "self":
                kwargs[key] = values[key]
        kwargs.update(paths)

        wandb.init(
            project="VWSD",
            config=kwargs,
            group="RunAlgorithms",
            tags=["zero-shot"],
        )

    for name, p in paths.items():

        if not (p.is_absolute() or p.is_relative_to(root)):
            p = root / p
        if name == "out_p":
            p.mkdir(parents=True)

        assert p.exists(), f"Cannot read {p}"
        paths[name] = p
        logger.info(f"Resolved Path {name} -> {p}")

    # load encoder
    encoder = build_clip_encoder(model_name=clip_model_name, device=device)

    # load data and clients
    vwsd_ds = VWSDDataset(
        dataframe_p=paths["vwsd_dataset_p"],
        images_root=paths["vwsd_images_p"],
        language=vwsd_language,
        translate_to_en=translate_vwsd,
    )  # type: ignore
    wc = WikiClient(summaries_df_p=paths["wiki_summaries_p"])  # type: ignore
    dc = DepCCClient(es_user="reader", es_pw="reader")
    vs = get_visualsem_dataset(paths["visualsem_dataset_p"])  # type: ignore
    vsc = VisualSemClient(
        clip_encoder=encoder,
        vs=vs,
        vs_embeddings_path=paths["visualsem_embeddings_p"],
        langs=[gloss_lang],
        normalize_embeddings=normalize_embeddings,
        init_all=False,
    )

    algos: Dict[str, Algorithm] = dict()

    for algo in run_algo:
        if algo == "nobg":
            algos[algo] = NoBackground(
                encoder=encoder,
                ds=vwsd_ds,
                normalize=normalize_embeddings,
                show_progress_bar=True,
            )
        elif algo == "wiki":
            algos[algo] = WikiGlosses(
                encoder=encoder,
                ds=vwsd_ds,
                wc=wc,
                normalize=normalize_embeddings,
                show_progress_bar=True,
            )
        elif algo == "depcc":
            algos[algo] = DepCC(
                encoder=encoder,
                ds=vwsd_ds,
                dc=dc,
                normalize=normalize_embeddings,
                show_progress_bar=True,
            )
        elif algo == "hyper":
            algos[algo] = Hypernym(
                encoder=encoder,
                ds=vwsd_ds,
                hypernym_df_p=paths["hypernym_df_p"],
                normalize=normalize_embeddings,
                show_progress_bar=True,
            )
        elif algo == "vsif":
            algos[algo] = VisualSemImageFirst(
                encoder=encoder,
                ds=vwsd_ds,
                vsc=vsc,
                normalize=normalize_embeddings,
                show_progress_bar=True,
            )
        elif algo == "vstf":
            #            algos[algo + "-t2t-avg-i2i-avg"] = VisualSemTextFirst(
            #                encoder=encoder,
            #                ds=vwsd_ds,
            #                vsc=vsc,
            #                t2i=False,
            #                single_embs=False,
            #                i2i_single_embs=False,
            #                normalize=normalize_embeddings,
            #                show_progress_bar=True,
            #            )
            #            algos[algo + "-t2t-avg-i2i-single"] = VisualSemTextFirst(
            #                encoder=encoder,
            #                ds=vwsd_ds,
            #                vsc=vsc,
            #                t2i=False,
            #                single_embs=False,
            #                i2i_single_embs=True,
            #                normalize=normalize_embeddings,
            #                show_progress_bar=True,
            #            )
            #            algos[algo + "-t2t-single-i2i-avg"] = VisualSemTextFirst(
            #                encoder=encoder,
            #                ds=vwsd_ds,
            #                vsc=vsc,
            #                t2i=False,
            #                single_embs=True,
            #                i2i_single_embs=False,
            #                normalize=normalize_embeddings,
            #                show_progress_bar=True,
            #            )
            #            algos[algo + "-t2t-single-i2i-single"] = VisualSemTextFirst(
            #                encoder=encoder,
            #                ds=vwsd_ds,
            #                vsc=vsc,
            #                t2i=False,
            #                single_embs=True,
            #                i2i_single_embs=True,
            #                normalize=normalize_embeddings,
            #                show_progress_bar=True,
            #            )
            #            algos[algo + "-t2i-single-i2i-avg"] = VisualSemTextFirst(
            #                encoder=encoder,
            #                ds=vwsd_ds,
            #                vsc=vsc,
            #                t2i=True,
            #                single_embs=True,
            #                i2i_single_embs=False,
            #                normalize=normalize_embeddings,
            #                show_progress_bar=True,
            #            )
            #            algos[algo + "-t2i-single-i2i-single"] = VisualSemTextFirst(
            #                encoder=encoder,
            #                ds=vwsd_ds,
            #                vsc=vsc,
            #                t2i=True,
            #                single_embs=True,
            #                i2i_single_embs=True,
            #                normalize=normalize_embeddings,
            #                show_progress_bar=True,
            #            )
            #            algos[algo + "-t2i-avg-i2i-single"] = VisualSemTextFirst(
            #                encoder=encoder,
            #                ds=vwsd_ds,
            #                vsc=vsc,
            #                t2i=True,
            #                single_embs=False,
            #                i2i_single_embs=True,
            #                normalize=normalize_embeddings,
            #                show_progress_bar=True,
            #            )

            # Results on test set:                          Hits@1 	Hits@3 	Hits@5 	Hits@10 	MRR
            # VisualSemTextFirst(t2i-AvgEmbs,i2i-AvgEmbs) 	0.675214 	0.846542 	0.914141 	1.0 	0.777908
            # VisualSemTextFirst(t2i-AvgEmbs,i2i-SingleEmbs) 	0.675214 	0.846542 	0.914141 	1.0 	0.777908
            # VisualSemTextFirst(t2i-SingleEmbs,i2i-AvgEmbs) 	0.639860 	0.824398 	0.895882 	1.0 	0.751270
            # VisualSemTextFirst(t2i-SingleEmbs,i2i-SingleEmbs) 	0.639860 	0.824398 	0.895882 	1.0 	0.751270
            # VisualSemTextFirst(t2t-SingleEmbs,i2i-AvgEmbs) 	0.573038 	0.785936 	0.871018 	1.0 	0.703476
            # VisualSemTextFirst(t2t-SingleEmbs,i2i-SingleEmbs) 	0.573038 	0.785936 	0.871018 	1.0 	0.703476
            # VisualSemTextFirst(t2t-AvgEmbs,i2i-AvgEmbs) 	0.152681 	0.385392 	0.608780 	1.0 	0.355743
            # VisualSemTextFirst(t2t-AvgEmbs,i2i-SingleEmbs) 	0.152681 	0.385392 	0.608780 	1.0 	0.355743
            algos[algo + "-t2i-avg-i2i-avg"] = VisualSemTextFirst(
                encoder=encoder,
                ds=vwsd_ds,
                vsc=vsc,
                t2i=True,
                single_embs=False,
                i2i_single_embs=False,
                normalize=normalize_embeddings,
                show_progress_bar=True,
            )

    results = []
    for _, algo in algos.items():
        logger.info(f"Running {algo.name} Algorithm...")
        # if algo not in ['nobg', 'vsif', 'vstf'] or not (vwsd_ds.language == "en" or vwsd_ds.translate_to_en):
        #    raise NotImplementedError()
        result_df = algo.run(batch_size=10, eval=False, persist_dir_p=paths["out_p"])
        if wandb_logging:
            wandb.log({f"{algo.name} Results": result_df})
        results.append(result_df)

        if vwsd_ds.has_gold:
            algo.eval(results=result_df, persist_dir_p=paths["out_p"])

    if vwsd_ds.has_gold:
        all_results = pd.concat(results)
        all_results = all_results.sort_values(by=["idx", "algo"]).reset_index(drop=True)
        all_scores = evaluate_algorithm_results(all_results)
        if wandb_logging:
            wandb.log({"All Algorithm Scores": all_scores})
        all_scores.to_parquet(paths["out_p"] / "all_algo_scores.df.pq")

    return paths["out_p"]


if __name__ == "__main__":
    fire.Fire(main)
