# Metrics

This page collects the metrics used to evaluate models in this repo — what each one actually measures, why it's worth looking at, and how it's computed, with the formula and a small runnable snippet for each.

They split into three groups:

- **Teacher-forced** — fed the gold token at every step. Cheap, deterministic, but optimistic (the model never has to recover from its own mistakes). Loss, perplexity, token/top-5 accuracy.
- **Generation** — the model decodes free-running, then we score the produced text against the reference. Slower and noisier, but it's the regime you actually deploy in. ROUGE, BLEU.
- **Efficiency** — wall-clock and memory. Throughput, latency, peak memory.

A recurring theme: teacher-forced metrics and generation metrics can disagree, so always read them together — see [Reading them together](#reading-them-together) below.

Notation used below: $N$ non-pad target tokens, vocabulary size $V$, logits $z_{i} \in \mathbb{R}^{V}$ for token $i$, gold label $y_i$.

---

## Cross-entropy loss (`eval_loss`)

**What.** The average negative log-likelihood the model assigns to the correct next token, over non-pad positions.

**Why.** It's the training objective itself, so it's the most direct read of how well the model fits the data. Lower is better. Because it's a smooth, per-token average it's far less noisy than the generation metrics — good for tracking small changes.

**Formula.**

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \log p_\theta(y_i \mid x_{<i}), \qquad p_\theta(y_i\mid x_{<i}) = \mathrm{softmax}(z_i)_{y_i}$$

Padding positions are excluded via `ignore_index`, and the repo sums the loss then divides by the real token count $N$ (`reduction="sum"` / `tokens`) so that variable-length batches are weighted by tokens, not by sequence.

```python
import torch
import torch.nn.functional as F

# logits: [N, V], labels: [N], pad_id excluded
logits = torch.tensor([[2.0, 0.5, 0.1], [0.1, 3.0, 0.2]])
labels = torch.tensor([0, 1])
pad_id = -100

loss = F.cross_entropy(logits, labels, ignore_index=pad_id, reduction="sum")
n_tokens = (labels != pad_id).sum()
eval_loss = (loss / n_tokens).item()
print(round(eval_loss, 4))   # 0.2132
```

---

## Perplexity (`perplexity`)

**What.** The exponential of the cross-entropy. Intuitively, the effective number of equally-likely choices the model is hesitating between at each step — perplexity 12 ≈ "as unsure as a fair 12-sided die".

**Why.** Same information as the loss but on an interpretable scale, and it's the number most LM papers report, so it makes comparison across models easy. Lower is better.

**Formula.**

$$\mathrm{PPL} = \exp(\mathcal{L})$$

The repo clamps to infinity when $\mathcal{L} > 88$ to avoid `exp` overflow on a diverged run.

```python
import math
eval_loss = 0.2132
ppl = math.inf if eval_loss > 88 else math.exp(eval_loss)
print(round(ppl, 4))   # 1.2376
```

---

## Token accuracy (`token_accuracy`)

**What.** Fraction of non-pad positions where the single highest-logit token is the correct one (top-1 / greedy-match accuracy).

**Why.** A loss can drop without the argmax improving, and vice versa, so accuracy is a complementary, threshold-free view of "would greedy decoding get this token right". Higher is better. Note it's still teacher-forced, so it doesn't tell you about compounding errors.

**Formula.**

$$\mathrm{Acc}_{1} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}\!\left[\arg\max_v z_{i,v} = y_i\right]$$

```python
import torch
logits = torch.tensor([[2.0, 0.5, 0.1], [0.1, 3.0, 0.2]])
labels = torch.tensor([0, 1])
correct = (logits.argmax(dim=-1) == labels).sum()
acc = (correct / labels.numel()).item()
print(acc)   # 1.0
```

---

## Top-5 accuracy (`top_5_accuracy`)

**What.** Fraction of positions where the gold token is among the model's 5 highest-logit tokens.

**Why.** Top-1 is harsh — there are often several reasonable next tokens. Top-5 measures whether the model is "in the right neighbourhood", which correlates better with how sampling/beam decoding behaves. Higher is better. (The repo uses $k=\min(5, V)$ so it's safe on tiny vocabularies.)

**Formula.**

$$\mathrm{Acc}_{5} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}\!\left[y_i \in \mathrm{top\text{-}5}(z_i)\right]$$

```python
import torch
logits = torch.tensor([[2.0, 0.5, 0.1, 1.7, 0.3, 0.9]])
labels = torch.tensor([3])
k = min(5, logits.shape[-1])
topk = logits.topk(k, dim=-1).indices            # [[0, 3, 5, 1, 4]]
hit = (topk == labels[:, None]).any(dim=-1)
print(hit.float().mean().item())   # 1.0
```

---

## ROUGE-1 / ROUGE-2 / ROUGE-L (`rouge1`, `rouge2`, `rougeL`)

**What.** Overlap between the generated summary and the reference. ROUGE-1 = unigram overlap, ROUGE-2 = bigram overlap, ROUGE-L = longest-common-subsequence overlap (rewards in-order matches without requiring them to be contiguous). The repo reports the **F-measure** of each, averaged over generated examples, with Porter stemming on.

**Why.** This is the standard summarization metric and — crucially — it scores *generated* text, so it exposes the teacher-forcing blind spot. Recall-leaning by design (did the summary cover the reference content), which fits summarization. Higher is better.

**Formula.** For ROUGE-N with reference and candidate n-gram multisets, with precision $P$ and recall $R$:

$$R = \frac{\sum_{g}\min(c_{\text{cand}}(g),\, c_{\text{ref}}(g))}{\sum_{g} c_{\text{ref}}(g)}, \quad P = \frac{\sum_{g}\min(c_{\text{cand}}(g),\, c_{\text{ref}}(g))}{\sum_{g} c_{\text{cand}}(g)}, \quad F_1 = \frac{2PR}{P+R}$$

ROUGE-L replaces the n-gram count with the length of the longest common subsequence (LCS) between candidate and reference token sequences.

```python
# pip install rouge-score
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
ref  = "the council approved the new parking budget"
pred = "council approved a new budget for parking"
scores = scorer.score(ref, pred)
for k, v in scores.items():
    print(k, round(v.fmeasure, 3))
# rouge1 0.714
# rouge2 0.167
# rougeL 0.571
```

Across a set, the repo averages each metric's f-measure over all examples: $\mathrm{ROUGE} = \frac{1}{M}\sum_{j=1}^{M} F_1^{(j)}$ for $M$ generated summaries.

---

## BLEU (`bleu`)

**What.** Corpus-level n-gram precision (up to 4-grams) of the generations against the references, with a brevity penalty for output that's too short. Computed with `sacrebleu` so the tokenization is standardized and the score is comparable across projects. Range 0–100.

**Why.** Precision-leaning counterpart to ROUGE (did the summary avoid saying things that aren't in the reference). It's the most sensitive of the text metrics to fluency and phrasing, and often the first to move when generation quality changes. Higher is better.

**Formula.** With modified n-gram precisions $p_n$, weights $w_n = 1/4$, candidate length $c$, reference length $r$:

$$\mathrm{BLEU} = \underbrace{\min\!\left(1,\, e^{1 - r/c}\right)}_{\text{brevity penalty}} \cdot \exp\!\left(\sum_{n=1}^{4} w_n \log p_n\right)$$

```python
# pip install sacrebleu
import sacrebleu
preds = ["council approved a new budget for parking"]
refs  = [["the council approved the new parking budget"]]   # list-of-references
bleu = sacrebleu.corpus_bleu(preds, refs)
print(round(bleu.score, 2))   # 16.52
```

> BLEU is a *corpus* statistic — n-gram counts are pooled across all examples before the ratio is taken, so it's unreliable on one sentence and only meaningful over the full eval set.

---

## Throughput (`eval_tokens_per_sec`, `generation_tokens_per_sec`)

**What.** Tokens processed per wall-clock second. `eval_tokens_per_sec` covers the teacher-forced forward passes (highly parallel — every position scored at once); `generation_tokens_per_sec` covers autoregressive decoding (one token at a time, so always much lower).

**Why.** Quality and cost are a trade — an efficiency-oriented variant only wins if it holds quality *and* throughput. Reporting both prefill-style and decode-style throughput separates "fast to score" from "fast to generate". Higher is better.

**Formula.**

$$\text{tok/s} = \frac{\text{tokens processed}}{\text{elapsed seconds}}$$

```python
import time
n_tokens = 60_829
t0 = time.monotonic()
# ... run the forward / generation loop ...
elapsed = max(time.monotonic() - t0, 1e-9)   # guard against divide-by-zero
print(round(n_tokens / elapsed))
```

---

## Forward latency (`avg_forward_latency_ms`)

**What.** Mean time for a single evaluation forward pass (one batch), in milliseconds. On CUDA the repo calls `torch.cuda.synchronize()` before and after timing so the number reflects real device work, not just async kernel-launch time.

**Why.** Throughput hides batching effects; per-call latency is what a latency-bound serving path cares about. Lower is better.

**Formula.** For per-batch times $t_1,\dots,t_B$:

$$\overline{t}_{\text{ms}} = \frac{1000}{B}\sum_{b=1}^{B} t_b$$

```python
import time, torch

def timed_forward(model, batch, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.monotonic()
    with torch.inference_mode():
        model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)   # wait for kernels before stopping the clock
    return (time.monotonic() - t0) * 1000   # ms
```

---

## Peak CUDA memory (`peak_cuda_memory_mb`)

**What.** High-water mark of allocated GPU memory during evaluation, in MB (`None` on CPU). The counter is reset with `torch.cuda.reset_peak_memory_stats()` at the start of the eval loop.

**Why.** Memory is the hard constraint that decides batch size, context length, and whether a model fits at all — a method that's faster but blows the memory budget isn't usable. Lower is better.

**Formula.**

$$\text{peak MB} = \frac{\max_t \text{bytes allocated}(t)}{10^{6}}$$

```python
import torch
device = torch.device("cuda")
torch.cuda.reset_peak_memory_stats(device)
# ... run eval ...
peak_mb = torch.cuda.max_memory_allocated(device) / 1_000_000
print(round(peak_mb, 1))
```

---

## Reading them together

No single number is enough, and the teacher-forced and generation groups can point in opposite directions. A pattern worth recognising:

| signal | what it can look like |
|--------|-----------------------|
| loss / perplexity / token acc | model A marginally **better** — suggests a change did nothing, or even hurt |
| ROUGE-L, BLEU | model B clearly **better** — the change actually mattered |
| throughput / memory | unchanged — no trade |

This happens because teacher forcing feeds the gold token at every step, so the model never has to recover from its own mistakes — a generation-time problem can stay completely hidden in the loss. If you only ever watch perplexity, you miss it. Rule of thumb for this repo: **track loss for training health, but judge a model on generation.**
