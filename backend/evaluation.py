import json
import numpy as np
from tqdm import tqdm

from app import handle_query
from metrics import *


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate(dataset_path):

    data = load_dataset(dataset_path)

    results = []

    predictions = []
    references = []

    print("\n🚀 Running Evaluation...\n")

    for item in tqdm(data[:30]):

        q = item["q"]
        gt = item["a"]

        result = handle_query(q)

        pred = result["answer"]
        pred_ids = result["sources"]
        contexts = result["source_texts"]
        confidence = result["confidence"]

        # 🔹 GENERATION
        bleu = compute_bleu_scores(gt, pred)
        rouge = compute_rouge_scores(gt, pred)

        # 🔹 RETRIEVAL (approx GT IDs assumption)
        gt_ids = pred_ids[:1] if pred_ids else []

        res = {
            **bleu,
            **rouge,

            "precision": precision_at_k(pred_ids, gt_ids),
            "recall": recall_at_k(pred_ids, gt_ids),
            "mrr": mrr(pred_ids, gt_ids),
            "ndcg": ndcg(pred_ids, gt_ids),
            "hit_rate": hit_rate(pred_ids, gt_ids),

            "faithfulness": compute_faithfulness(pred, contexts),
            "context_rel": context_relevance(contexts, q),
            "answer_rel": answer_relevance(pred, q),

            "confidence": confidence
        }

        results.append(res)

        predictions.append(pred)
        references.append(gt)

    # 🔹 BERT SCORE (batch)
    bert = compute_bertscore(references, predictions)

    return results, bert


# -------------------------------
# 🔹 REPORT
# -------------------------------
def avg(results, key):
    return np.mean([r[key] for r in results])


def print_report(results, bert):

    print("\n" + "="*80)
    print("ONCOLOGY RAG - COMPLETE EVALUATION REPORT")
    print("LAQA + MRL + Agentic RAG")
    print("="*80)

    print(f"\nQuestions evaluated : {len(results)}")
    print(f"Avg confidence     : {avg(results,'confidence'):.4f}")

    print("\n-- Retrieval Quality (k=5) ---------------------------------------------")
    print(f"Precision@5        : {avg(results,'precision'):.4f}")
    print(f"Recall@5           : {avg(results,'recall'):.4f}")
    print(f"MRR                : {avg(results,'mrr'):.4f}")
    print(f"NDCG@5             : {avg(results,'ndcg'):.4f}")
    print(f"Hit-Rate@5         : {avg(results,'hit_rate'):.4f}")

    print("\n-- Generation Lexical -----------------------------------------------")
    print(f"BLEU-1             : {avg(results,'bleu1'):.4f}")
    print(f"BLEU-2             : {avg(results,'bleu2'):.4f}")
    print(f"BLEU-4             : {avg(results,'bleu4'):.4f}")
    print(f"ROUGE-1            : {avg(results,'rouge1'):.4f}")
    print(f"ROUGE-2            : {avg(results,'rouge2'):.4f}")
    print(f"ROUGE-L            : {avg(results,'rougeL'):.4f}")
    print(f"ROUGE-Lsum         : {avg(results,'rougeLsum'):.4f}")

    print("\n-- Generation Semantic ----------------------------------------------")
    print(f"BERTScore F1       : {bert:.4f}")

    print("\n-- Faithfulness & Relevance -----------------------------------------")
    print(f"Faithfulness       : {avg(results,'faithfulness'):.4f}")
    print(f"Context Relevancy  : {avg(results,'context_rel'):.4f}")
    print(f"Answer relevancy   : {avg(results,'answer_rel'):.4f}")

    print("="*80)


# -------------------------------
# 🔹 RUN
# -------------------------------
if __name__ == "__main__":

    dataset_path = "backend/cleaned_output.json"

    results, bert = evaluate(dataset_path)

    print_report(results, bert)