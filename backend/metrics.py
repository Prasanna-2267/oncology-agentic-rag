import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score

smooth = SmoothingFunction().method1
rouge = rouge_scorer.RougeScorer(
    ['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
    use_stemmer=True
)


# -------------------------------
# 🔹 GENERATION METRICS
# -------------------------------
def compute_bleu_scores(ref, pred):
    ref_tokens = ref.split()
    pred_tokens = pred.split()

    return {
        "bleu1": sentence_bleu([ref_tokens], pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth),
        "bleu2": sentence_bleu([ref_tokens], pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth),
        "bleu4": sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth),
    }


def compute_rouge_scores(ref, pred):
    scores = rouge.score(ref, pred)
    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
        "rougeLsum": scores["rougeLsum"].fmeasure
    }


def compute_bertscore(refs, preds):
    P, R, F1 = bert_score(preds, refs, lang="en", verbose=False)
    return float(F1.mean())


# -------------------------------
# 🔹 RETRIEVAL METRICS
# -------------------------------
def precision_at_k(pred_ids, gt_ids, k=5):
    return len(set(pred_ids[:k]) & set(gt_ids)) / k


def recall_at_k(pred_ids, gt_ids, k=5):
    if not gt_ids:
        return 0
    return len(set(pred_ids[:k]) & set(gt_ids)) / len(gt_ids)


def mrr(pred_ids, gt_ids):
    for i, pid in enumerate(pred_ids):
        if pid in gt_ids:
            return 1 / (i + 1)
    return 0


def hit_rate(pred_ids, gt_ids):
    return int(any(pid in gt_ids for pid in pred_ids))


def ndcg(pred_ids, gt_ids, k=5):
    dcg = 0
    for i, pid in enumerate(pred_ids[:k]):
        if pid in gt_ids:
            dcg += 1 / np.log2(i + 2)

    ideal = sum(1 / np.log2(i + 2) for i in range(min(len(gt_ids), k)))
    return dcg / ideal if ideal > 0 else 0


# -------------------------------
# 🔹 FAITHFULNESS (REAL)
# -------------------------------
def compute_faithfulness(answer, contexts):
    context_text = " ".join(contexts).lower()
    answer_words = set(answer.lower().split())

    if not answer_words:
        return 0

    overlap = sum(1 for w in answer_words if w in context_text)
    return overlap / len(answer_words)


# -------------------------------
# 🔹 RELEVANCE (REAL)
# -------------------------------
def answer_relevance(answer, question):
    q_words = set(question.lower().split())
    a_words = set(answer.lower().split())

    if not q_words:
        return 0

    overlap = len(q_words & a_words)
    return overlap / len(q_words)


def context_relevance(contexts, question):
    context_text = " ".join(contexts).lower()
    q_words = set(question.lower().split())

    if not q_words:
        return 0

    overlap = sum(1 for w in q_words if w in context_text)
    return overlap / len(q_words)