# Dataset and tokenization variants

Datasets are registered through `src.registry.DATASET`. They return dataloaders, tokenizers, vocab sizes, and pad/eos ids in the shape expected by the trainer.

## Summarization datasets

### `meetingbank`

Dataset tag: [MeetingBank](https://huggingface.co/datasets/huuuyeah/meetingbank)

Config: [`meetingbank.yaml`](../configs/data/meetingbank.yaml)
Implementation: [`meetingbank.py`](../src/components/datasets/meetingbank.py)

This path builds a source/target summarization dataset. Source transcripts are truncated to `max_src_len`; targets are truncated to `max_tgt_len`. The tokenizer is cached under `tokenizer_dir`.

Use it for encoder-decoder meeting summarization, or for causal summarization experiments through the causal MeetingBank path.

### `multi_news`

Dataset tag: [Multi-News](https://huggingface.co/datasets/alexfabbri/multi_news)

Config: [`multi_news.yaml`](../configs/data/multi_news.yaml)
Implementation: [`multi_news.py`](../src/components/datasets/multi_news.py)

This path uses multi-document news inputs and summaries. It shares the source/target summarization shape with MeetingBank but has larger default sequence limits.

Use it when testing whether a component change transfers beyond meeting transcripts.

## Causal pretraining datasets

Implementation: [`fineweb_edu.py`](../src/components/datasets/fineweb_edu.py)

The causal dataset path streams text, trains or loads a byte-level BPE tokenizer, appends `[EOS]`, and packs text into fixed `(seq_len + 1)` chunks:

\[
x = tokens_{0:n-1},\quad y = tokens_{1:n}
\]

### `fineweb_edu`

Dataset tag: [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)

Config: [`fineweb_edu.yaml`](../configs/data/fineweb_edu.yaml)

Default HF path is `HuggingFaceFW/fineweb-edu`, config `sample-10BT`.

### `fineweb_edu_100bt`

Dataset tag: [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)

Config: [`fineweb_edu_100bt.yaml`](../configs/data/fineweb_edu_100bt.yaml)

Same builder as `fineweb_edu`, but points at `sample-100BT` and a separate tokenizer cache basename.

### `c4`

Dataset tag: [C4](https://huggingface.co/datasets/allenai/c4)

Config: [`c4.yaml`](../configs/data/c4.yaml)

Uses the same packed causal-LM builder with HF path `allenai/c4`, config `en`.

### `wikitext`

Dataset tag: [WikiText-103](https://huggingface.co/datasets/Salesforce/wikitext)

Config: [`wikitext.yaml`](../configs/data/wikitext.yaml)

Uses `Salesforce/wikitext`, config `wikitext-103-raw-v1`, with a validation split.

### `wikipedia`

Dataset tag: [Wikimedia Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia)

Config: [`wikipedia.yaml`](../configs/data/wikipedia.yaml)

Uses the same packed causal-LM path with HF config `20231101.en`.
