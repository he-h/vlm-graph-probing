from typing import List
from aac_metrics.functional import spice, meteor, rouge_l, bleu
from bert_score import score as bertscore_score
from pycocoevalcap.meteor.meteor import Meteor
import torch
import re
import unicodedata as ud

def sanitize(s: str) -> str:
    """Normalize and remove newline, carriage return, and tabs."""
    if not isinstance(s, str):
        s = str(s)
    s = ud.normalize("NFKC", s)   # normalize unicode
    s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def sanitize_preds_refs(preds: List[str], refs: List[List[str]]) -> (List[str], List[List[str]]):
    """Sanitize predictions and references."""
    preds = [sanitize(p) for p in preds]
    refs = [[sanitize(r) for r in ref_list] for ref_list in refs]
    return preds, refs


def spice_scores(preds: List[str], refs: List[List[str]]) -> List[float]:
    """Calculate SPICE scores for raw strings."""
    _, sent = spice(preds, refs, cache_path=None, java_max_memory="8G")
    return sent["spice"].tolist()


def meteor_scores(preds: List[str], refs: List[List[str]]) -> List[float]:
    """Calculate METEOR scores for raw strings."""
    _, sent = meteor(preds, refs)
    return sent["meteor"].tolist()


def rougeL_scores(preds: List[str], refs: List[List[str]]) -> List[float]:
    """Calculate ROUGE-L scores for raw strings."""
    _, sent = rouge_l(preds, refs)
    return sent["rouge_l"].tolist()


def bleu4_scores(preds: List[str], refs: List[List[str]]) -> List[float]:
    """Calculate BLEU-4 scores for raw strings."""
    _, sent = bleu(preds, refs, n=4)
    return sent["bleu_4"].tolist()


def bertscore_f1(
    preds: List[str], 
    refs: List[List[str]], 
    model_type: str = "roberta-large"
) -> List[float]:
    """Per-sample BERTScore F1 (best among references)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    out = []
    for cand, ref_list in zip(preds, refs):
        cands = [cand] * len(ref_list)
        P, R, F1 = bertscore_score(
            cands=cands,
            refs=ref_list,
            model_type=model_type,
            lang="en",
            rescale_with_baseline=True,
            verbose=False,
            device=device
        )
        out.append(float(F1.max().item()))
    
    return out


if __name__ == "__main__":
    # Test data
    preds = [
        "A man rides a bike on the road.",
        "A dog plays with a frisbee on the beach."
    ]
    refs = [
        ["A man is riding a bicycle on the street.", "A person rides a bike down a road."],
        ["A dog is playing with a disc on the shore.", "A canine plays with a frisbee at the beach."]
    ]
    
    print("SPICE:", spice_scores(preds, refs))
    print("METEOR:", meteor_scores(preds, refs))
    print("ROUGE-L:", rougeL_scores(preds, refs))
    print("BLEU-4:", bleu4_scores(preds, refs))
    print("BERTScore F1:", bertscore_f1(preds, refs))