import argparse
from collections import defaultdict
from typing import Optional
from dataclasses import dataclass, field

from sentence_transformers import CrossEncoder
import torch
from transformers import HfArgumentParser
import numpy as np

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


class BaseCrossEncoder:
    def __init__(self, model, batch_size=32, device="cuda"):
        self.model = model
        self.batch_size = batch_size
        self.model.to(device)

    def _passage_to_string(self, doc_item):
        if "document" not in doc_item:
            content = doc_item['contents']
        else:
            content = doc_item['document']['contents']
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])

        return f"(Title: {title}) {text}"

    def rerank(self,
               queries: list[str],
               documents: list[list[dict]],
               return_doc_item: bool = True):
        """
        Rerank documents for each query.

        Args:
            queries: List of query strings
            documents: List of document lists (one per query), where each doc is a dict with "id" and "contents"
            return_doc_item: If True, return (doc_string, score, doc_item). If False, return (doc_string, score) for backward compatibility.

        Returns:
            Dict mapping query_id to sorted list of tuples:
            - If return_doc_item=True: (doc_string, score, doc_item)
            - If return_doc_item=False: (doc_string, score)
        """
        assert len(queries) == len(documents)

        pairs = []
        qids = []
        doc_items = []  # Store original doc_items for later use
        for qid, (query, doc_list) in enumerate(zip(queries, documents)):
            for doc_item in doc_list:
                doc = self._passage_to_string(doc_item)
                pairs.append((query, doc))
                qids.append(qid)
                doc_items.append(doc_item)  # Keep original structure

        scores = self._predict(pairs)
        query_to_doc_scores = defaultdict(list)

        assert len(scores) == len(pairs) == len(qids) == len(doc_items)
        for i in range(len(pairs)):
            query, doc = pairs[i]
            score = scores[i]
            qid = qids[i]
            doc_item = doc_items[i]
            # Store based on return_doc_item flag
            if return_doc_item:
                query_to_doc_scores[qid].append((doc, score, doc_item))
            else:
                # Backward compatibility: return only (doc, score)
                query_to_doc_scores[qid].append((doc, score))

        sorted_query_to_doc_scores = {}
        for query, doc_scores in query_to_doc_scores.items():
            sorted_query_to_doc_scores[query] = sorted(doc_scores, key=lambda x: x[1], reverse=True)

        return sorted_query_to_doc_scores

    def _predict(self, pairs: list[tuple[str, str]]):
        raise NotImplementedError 

    @classmethod
    def load(cls, model_name_or_path, **kwargs):
        raise NotImplementedError


class SentenceTransformerCrossEncoder(BaseCrossEncoder):
    def __init__(self, model, batch_size=32, device="cuda"):
        super().__init__(model, batch_size, device)

    def _predict(self, pairs: list[tuple[str, str]]):
        scores = self.model.predict(pairs, batch_size=self.batch_size)
        scores = scores.tolist() if isinstance(scores, torch.Tensor) or isinstance(scores, np.ndarray) else scores
        return scores

    @classmethod
    def load(cls, model_name_or_path, **kwargs):
        model = CrossEncoder(model_name_or_path)
        return cls(model, **kwargs)


class RerankRequest(BaseModel):
    queries: list[str]
    documents: list[list[dict]]
    rerank_topk: Optional[int] = None
    return_scores: bool = False


@dataclass 
class RerankerArguments:
    max_length: int = field(default=512)
    rerank_topk: int = field(default=3)
    rerank_model_name_or_path: str = field(default="cross-encoder/ms-marco-MiniLM-L12-v2")
    batch_size: int = field(default=32)
    reranker_type: str = field(default="sentence_transformer")

def get_reranker(config):
    if config.reranker_type == "sentence_transformer":
        return SentenceTransformerCrossEncoder.load(
            config.rerank_model_name_or_path,
            batch_size=config.batch_size,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        raise ValueError(f"Unknown reranker type: {config.reranker_type}")


app = FastAPI()

@app.post("/rerank")
def rerank_endpoint(request: RerankRequest):
    """
    Endpoint that accepts queries and performs retrieval.
    Input format:
    {
      "queries": ["What is Python?", "Tell me about neural networks."],
      "documents": [[doc_item_1, ..., doc_item_k], [doc_item_1, ..., doc_item_k]],
      "rerank_topk": 3,
      "return_scores": true
    }
    """
    if not request.rerank_topk:
        request.rerank_topk = config.rerank_topk  # fallback to default

    # Perform batch re reranking
    # doc_scores already sorted by score
    query_to_doc_scores = reranker.rerank(request.queries, request.documents) 

    # Format response
    resp = []
    for _, doc_scores in query_to_doc_scores.items():
        doc_scores = doc_scores[:request.rerank_topk]
        if request.return_scores:
            combined = []
            # Now doc_scores contains (doc_string, score, doc_item) tuples
            for doc, score, doc_item in doc_scores:
                combined.append({
                    "document": doc,
                    "score": score,
                    "doc_id": doc_item.get('id', None)  # Include doc ID if available
                })
            resp.append(combined)
        else:
            # Extract doc_string from (doc_string, score, doc_item) tuples
            resp.append([doc for doc, _, _ in doc_scores])
    return {"result": resp}


if __name__ == "__main__":
    
    # 1) Build a config (could also parse from arguments).
    #    In real usage, you'd parse your CLI arguments or environment variables.
    parser = HfArgumentParser((RerankerArguments))
    config = parser.parse_args_into_dataclasses()[0]

    # 2) Instantiate a global retriever so it is loaded once and reused.
    reranker = get_reranker(config)
    
    # 3) Launch the server. By default, it listens on http://127.0.0.1:8000
    uvicorn.run(app, host="0.0.0.0", port=6980)
