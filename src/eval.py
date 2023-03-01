import pandas as pd
import numpy as np


def evaluate_algorithm_results(algo_results: pd.DataFrame) -> pd.DataFrame:
    algo_results["hit_rank"] = algo_results.apply(
        lambda r: int(np.where(r["ranks"] == r["gold"])[0]), axis="columns"
    )

    algo_results["h@1"] = algo_results.apply(
        lambda r: r["hit_rank"] == 0, axis="columns"
    )
    algo_results["h@3"] = algo_results.apply(
        lambda r: r["hit_rank"] < 3, axis="columns"
    )
    algo_results["h@5"] = algo_results.apply(
        lambda r: r["hit_rank"] < 5, axis="columns"
    )
    algo_results["h@10"] = algo_results.apply(
        lambda r: r["hit_rank"] < 10, axis="columns"
    )

    scores = dict()
    algos = pd.unique(algo_results.algo)

    for algo in algos:
        algo_res = algo_results[algo_results.algo == algo]
        scores[algo] = dict()
        for i in [1, 3, 5, 10]:
            scores[algo][f"Hits@{i}"] = algo_res[f"h@{i}"].mean()
        scores[algo]["MRR"] = (1 / (algo_res["hit_rank"] + 1)).mean()

    return (
        pd.DataFrame.from_dict(scores).T
        .sort_values(by="Hits@1", ascending=False)
        .reindex(columns=[f"Hits@{i}" for i in [1, 3, 5, 10]] + ["MRR"])
    )
