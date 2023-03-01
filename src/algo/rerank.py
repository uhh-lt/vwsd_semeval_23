from abc import ABC, abstractmethod
from loguru import logger
import pandas as pd
import numpy as np
from scipy.special import softmax
from typing import List, Tuple

from src.eval import evaluate_algorithm_results


class ReRanker(ABC):
    @abstractmethod
    def __init__(self, name: str, algo_results: pd.DataFrame) -> None:

        assert all(
            c in algo_results.columns for c in ["idx", "sims", "ranks", "gold", "algo"]
        )
        self.algo_results = algo_results.sort_values(by=["idx", "algo"]).reset_index(
            drop=True
        )

        self.num_cols = len(algo_results.ranks[0])
        self.algos = pd.unique(algo_results.algo).tolist()
        self.num_algos = len(self.algos)

        assert len(self.algo_results) % self.num_algos == 0

        # reshape to group votes / rankings of algos per sample
        rankings = np.stack(self.algo_results["ranks"])
        self.rankings_per_sample = rankings.reshape(-1, self.num_algos, self.num_cols)

        self.name = name
        logger.info(
            f"Instantiated ReRanker '{self.name}' with results of the following algorithms {self.algos}!"
        )

    @abstractmethod
    def reranking_method(self) -> np.ndarray:
        pass

    def rerank(
        self, eval: bool = True
    ) -> pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame]:
        reranked = self.reranking_method()
        assert reranked.shape[0] == len(self.algo_results) / self.num_algos

        # create a new dataframe but make sure sample idx and gold are matching!
        res = (
            self.algo_results[["idx", "gold"]][:: self.num_algos]
            .reset_index(drop=True)
            .copy()
        )

        # fake sims
        fake_sims = softmax(np.arange(self.num_cols + 1)[::-1] / self.num_cols)
        assert np.isclose(np.sum(fake_sims), 1.0)
        res["sims"] = pd.Series(np.tile(fake_sims, (len(reranked), 1)).tolist()).apply(
            np.array
        )

        # voted rankings
        res["ranks"] = pd.Series(reranked.tolist()).apply(np.array)

        # name of the reranking
        res["algo"] = f"{self.name}"

        if eval:
            scores = evaluate_algorithm_results(res)
            return res, scores

        return res


class MajorityVoteReRanker(ReRanker):
    def __init__(
        self, algo_results: pd.DataFrame, break_ties: str | List = "rand"
    ) -> None:
        if break_ties not in ["min", "max", "rand"] and not isinstance(
            break_ties, (np.ndarray, List)
        ):
            raise ValueError(f'Incorrect BreakTies Argument: {break_ties}')
        super().__init__(
            f"MajorityVoteReRanker-BreakTies{break_ties.capitalize() if isinstance(break_ties, str) else 'AlgorithmOrdering' + str(break_ties) }",
            algo_results,
        )

        self.break_ties = break_ties

    def reranking_method(self) -> np.ndarray:
        # majority voting
        voted_rankings = np.array(
            [self.mode_break_ties(ranks) for ranks in self.rankings_per_sample]
        )
        return voted_rankings

    def mode_break_ties(self, votes: np.ndarray):
        assert votes.ndim == 2

        num_rows, num_cols = votes.shape
        todo = set(range(num_cols))
        out = np.zeros(num_cols, dtype=np.int64)

        for col in range(num_cols):
            # count the votes for the column
            counts = np.zeros(votes[:, col].max() + 1, dtype=np.int64)
            for v in votes[:, col]:
                counts[v] += 1

            # find the max vote(s)
            maxs = np.where(counts == counts.max())[0]
            if self.break_ties == "rand":
                v = np.random.choice(maxs)
            elif self.break_ties == "min":
                v = np.min(maxs)
            elif self.break_ties == "max":
                v = np.max(maxs)
            elif isinstance(self.break_ties, (np.ndarray, List)):
                if len(maxs) == 1:
                    v = maxs[0]
                else:
                    # break by provided order (row / voter indices)
                    assert (
                        len(self.break_ties) == num_rows
                        and np.max(self.break_ties) <= num_rows
                    )
                    for row_idx in self.break_ties:
                        if votes[row_idx, col] in maxs:
                            v = votes[row_idx, col]
                            break

            # make sure the vote to add to the rankings is not yet in there. if so, choose another one!
            if v in todo:
                todo.remove(v)
                out[col] = v
            else:
                # the vote for that column is already in the ranking but cannot appear twice
                if len(todo.intersection(np.where(counts > 0)[0])) == 0:
                    # there is no non-zero vote in for column still in todo(not in the ranking). so pick random from todo
                    smaller = np.random.choice(list(todo))
                    out[col] = smaller
                    todo.remove(smaller)
                else:
                    if len(todo) == 1:
                        # if there is only one left in todo, take it!
                        smaller = todo.pop()
                        out[col] = smaller
                    else:
                        # find a smaller (or tie) vote for that column still todo
                        for smaller in np.argsort(counts)[::-1]:
                            if smaller in todo:
                                out[col] = smaller
                                todo.remove(smaller)
                                break

        assert len(todo) == 0 and set(out) == set(
            range(num_cols)
        ), f"todo: {todo} out: {out}"
        return out


class RankScoringReRanker(ReRanker):
    def __init__(
        self, algo_results: pd.DataFrame, scoring: str = "rr", tau: float = 1.0
    ) -> None:
        super().__init__(
            f"RankScoringReRanker-{scoring.capitalize()}Tau{tau}", algo_results
        )

        self.scoring = scoring
        self.tau = tau

        if scoring == "rr":
            # reciprocal rank scoring
            self.rank_scores = 1 - np.arange(self.num_cols) / self.num_cols
        elif scoring == "softmax":
            # softmax rank scoring
            self.rank_scores = softmax(np.arange(self.num_cols)[::-1] / tau)
        elif scoring == "exp":
            # exponential decay scoring
            self.rank_scores = np.exp(-tau * np.arange(self.num_cols))
        else:
            raise ValueError

    def reranking_method(self) -> np.ndarray:
        reranked = []
        for ranks in self.rankings_per_sample:
            scores = np.zeros(self.num_cols)
            for rank in range(ranks.shape[1]):
                for vote in ranks[:, rank]:
                    scores[vote] += self.rank_scores[rank]
            reranked.append(np.argsort(scores)[::-1])
        reranked = np.stack(reranked)

        return reranked
