from typing import Dict, List

from elasticsearch6 import Elasticsearch

from src.vwsd_dataset import VWSDSampleBase
import re


class DepCCClient:
    def __init__(self,
                 es_user: str,
                 es_pw: str,
                 es_host: str = "ltheadnode.informatik.uni-hamburg.de:9200",
                 index: str = "depcc") -> None:

        self.es_client = Elasticsearch(hosts=f"{es_user}:{es_pw}@{es_host}")
        self.index = index
        self.cache: Dict[str, List[Dict[str, str]]] = dict()

    def _search(self, query: str, doc_id: bool = False) -> List[Dict[str, str]]:
        if query in self.cache:
            return self.cache[query]
        source = ["text"]
        if doc_id:
            source.append("document_id")
        res = self.es_client.search(index=self.index, body={
            "query": {
                "match": {
                    "text": {
                        "query": f"{query}",
                        "operator": "and"
                    }
                }
            },
            "_source": source
        })

        res = res["hits"]["hits"]
        ret = list(map(lambda h: h["_source"], res))
        self.cache[query] = ret
        return ret

    def _clean_str(self, inp: str) -> str:
        # remove multiple whitespaces
        out = re.sub(r'\s\s+', ' ', inp)

        # remove multiple dots
        out = re.sub(r'\.\.+', '.', out)

        return out

    def get_best_matching_longest_contexts(self, sample: VWSDSampleBase, max_sents: int = 5) -> List[str]:
        res = self._search(query=sample.context, doc_id=False)
        sents = sorted(map(lambda r: self._clean_str(r["text"]), res), key=lambda s: len(s), reverse=True)
        # remove exact duplicates
        return list(set(sents[:max_sents]))
