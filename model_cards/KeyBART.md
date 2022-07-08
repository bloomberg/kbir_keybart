# KeyBART
KeyBART as described in Learning Rich Representations of Keyphrase from Text (https://arxiv.org/pdf/2112.08547.pdf), pre-trains a BART-based architecture to produce a concatenated sequence of keyphrases in the CatSeqD format.

We provide some examples on Downstream Evaluations setups and and also how it can be used for Text-to-Text Generation in a zero-shot setting.

## Downstream Evaluation

### Keyphrase Generation
```
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("bloomberg/KeyBART")
model = AutoModelForSeq2SeqLM.from_pretrained("bloomberg/KeyBART")

from datasets import load_dataset

dataset = load_dataset("midas/kp20k")
```

Reported Results:

#### Present Keyphrase Generation
|               | Inspec |       | NUS   |       | Krapivin |       | SemEval |       | KP20k |       |
|---------------|--------|-------|-------|-------|----------|-------|---------|-------|-------|-------|
| Model         | F1@5   | F1@M  | F1@5  | F1@M  | F1@5     | F1@M  | F1@5    | F1@M  | F1@5  | F1@M  |
| catSeq        | 22.5   | 26.2  | 32.3  | 39.7  | 26.9     | 35.4  | 24.2    | 28.3  | 29.1  | 36.7  |
| catSeqTG      | 22.9   | 27    | 32.5  | 39.3  | 28.2     | 36.6  | 24.6    | 29.0  | 29.2  | 36.6  |
| catSeqTG-2RF1 | 25.3   | 30.1  | 37.5  | 43.3  | 30       | 36.9  | 28.7    | 32.9  | 32.1  | 38.6  |
| GANMR         | 25.8   | 29.9  | 34.8  | 41.7  | 28.8     | 36.9  | N/A     | N/A   | 30.3  | 37.8  |
| ExHiRD-h      | 25.3   | 29.1  | N/A   | N/A   | 28.6     | 34.7  | 28.4    | 33.5  | 31.1  | 37.4  |
| Transformer (Ye et al., 2021)  | 28.15  | 32.56 | 37.07 | 41.91 | 31.58    | 36.55 | 28.71   | 32.52 | 33.21 | 37.71 |
| BART*         | 23.59  | 28.46 | 35.00 | 42.65 | 26.91    | 35.37 | 26.72   | 31.91 | 29.25 | 37.51 |
| KeyBART-DOC*  | 24.42  | 29.57 | 31.37 | 39.24 | 24.21    | 32.60 | 24.69   | 30.50 | 28.82 | 37.59 |
| KeyBART*      | 24.49  | 29.69 | 34.77 | 43.57 | 29.24    | 38.62 | 27.47   | 33.54 | 30.71 | 39.76 |
| KeyBART* (Zero-shot)     | 30.72  | 36.89 | 18.86 | 21.67 | 18.35    | 20.46 | 20.25   | 25.82 | 12.57 | 15.41 |

#### Absent Keyphrase Generation
|               | Inspec |      | NUS  |      | Krapivin |      | SemEval |      | KP20k |      |
|---------------|--------|------|------|------|----------|------|---------|------|-------|------|
| Model         | F1@5   | F1@M | F1@5 | F1@M | F1@5     | F1@M | F1@5    | F1@M | F1@5  | F1@M |
| catSeq        | 0.4    | 0.8  | 1.6  | 2.8  | 1.8      | 3.6  | 1.6     | 2.8  | 1.5   | 3.2  |
| catSeqTG      | 0.5    | 1.1  | 1.1  | 1.8  | 1.8      | 3.4  | 1.1     | 1.8  | 1.5   | 3.2  |
| catSeqTG-2RF1 | 1.2    | 2.1  | 1.9  | 3.1  | 3.0      | 5.3  | 2.1     | 3.0  | 2.7   | 5.0  |
| GANMR         | 1.3    | 1.9  | 2.6  | 3.8  | 4.2      | 5.7  | N/A     | N/A  | 3.2   | 4.5  |
| ExHiRD-h      | 1.1    | 2.2  | N/A  | N/A  | 2.2      | 4.3  | 1.7     | 2.5  | 1.6   | 3.2  |
| Transformer  (Ye et al., 2021)  | 1.02   | 1.94 | 2.82 | 4.82 | 3.21     | 6.04 | 2.05    | 2.33 | 2.31  | 4.61 |
| BART*         | 1.08   | 1.96 | 1.80 | 2.75 | 2.59     | 4.91 | 1.34    | 1.75 | 1.77  | 3.56 |
| KeyBART-DOC*  | 0.99   | 2.03 | 1.39 | 2.74 | 2.40     | 4.58 | 1.07    | 1.39 | 1.69  | 3.38 |
| KeyBART*      | 0.95   | 1.81 | 1.23 | 1.90 | 3.09     | 6.08 | 1.96    | 2.65 | 2.03  | 4.26 |
| KeyBART*  (Zero-shot)     | 1.83   | 2.92 | 1.46 | 2.19 | 1.29     | 2.09 | 1.12    | 1.45 | 0.70  | 1.14 |


### Abstractive Summarization
```
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("bloomberg/KeyBART")
model = AutoModelForSeq2SeqLM.from_pretrained("bloomberg/KeyBART")

from datasets import load_dataset

dataset = load_dataset("cnn_dailymail")
```

Reported Results:

| Model        | R1    | R2    | RL    |
|--------------|-------|-------|-------|
| BART (Lewis et al., 2019)        | 44.16 | 21.28 | 40.9  |
| BART*        | 42.93 | 20.12 | 39.72 |
| KeyBART-DOC* | 42.92 | 20.07 | 39.69 |
| KeyBART*     | 43.10 | 20.26 | 39.90 |

## Zero-shot settings
```
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("bloomberg/KeyBART")
model = AutoModelForSeq2SeqLM.from_pretrained("bloomberg/KeyBART")
```

Alternatively use the Hosted Inference API console provided in https://huggingface.co/bloomberg/KeyBART

Sample Zero Shot result:

```
Input: In this work, we explore how to learn task specific language models aimed towards learning rich representation of keyphrases from text documents. 
We experiment with different masking strategies for pre-training transformer language models (LMs) in discriminative as well as generative settings. 
In the discriminative setting, we introduce a new pre-training objective - Keyphrase Boundary Infilling with Replacement (KBIR), 
showing large gains in performance (upto 9.26 points in F1) over SOTA, when LM pre-trained using KBIR is fine-tuned for the task of keyphrase extraction.
In the generative setting, we introduce a new pre-training setup for BART - KeyBART, that reproduces the keyphrases related to the input text in the CatSeq
format, instead of the denoised original input. This also led to gains in performance (upto 4.33 points in F1@M) over SOTA for keyphrase generation.
Additionally, we also fine-tune the pre-trained language models on named entity recognition (NER), question answering (QA), relation extraction (RE),
abstractive summarization and achieve comparable performance with that of the SOTA, showing that learning rich representation of keyphrases is indeed beneficial
for many other fundamental NLP tasks.

Output: language model;keyphrase generation;new pre-training objective;pre-training setup;

```
