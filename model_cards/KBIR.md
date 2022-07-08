# Keyphrase Boundary Infilling with Replacement (KBIR)
The KBIR model as described in Learning Rich Representations of Keyphrases from Text (https://arxiv.org/pdf/2112.08547.pdf) builds on top of the RoBERTa architecture by adding an Infilling head and a Replacement Classification head that is used during pre-training. However, these heads are not used during the downstream evaluation of the model and we only leverage the pre-trained embeddings. Discarding the heads thereby allows us to be compatible with all AutoModel classes that RoBERTa supports.

We provide examples on how to perform downstream evaluation on some of the tasks reported in the paper.
## Downstream Evaluation

### Keyphrase Extraction
```
from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("bloomberg/KBIR")
model = AutoModelForTokenClassification.from_pretrained("bloomberg/KBIR")

from datasets import load_dataset

dataset = load_dataset("midas/semeval2017_ke_tagged")
```

Reported Results:

| Model                 | Inspec | SE10  | SE17  |
|-----------------------|--------|-------|-------|
| RoBERTa+BiLSTM-CRF    | 59.5   | 27.8  | 50.8  |
| RoBERTa+TG-CRF        | 60.4   | 29.7  | 52.1  |
| SciBERT+Hypernet-CRF  | 62.1   | 36.7  | 54.4  |
| RoBERTa+Hypernet-CRF  | 62.3   | 34.8  | 53.3  |
| RoBERTa-extended-CRF* | 62.09  | 40.61 | 52.32 |
| KBI-CRF*              | 62.61  | 40.81 | 59.7  |
| KBIR-CRF*             | 62.72  | 40.15 | 62.56 |

### Named Entity Recognition
```
from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("bloomberg/KBIR")
model = AutoModelForTokenClassification.from_pretrained("bloomberg/KBIR")

from datasets import load_dataset

dataset = load_dataset("conll2003")
```

Reported Results:

| Model                           | F1    |
|---------------------------------|-------|
| LSTM-CRF (Lample et al., 2016)  | 91.0  |
| ELMo (Peters et al., 2018)      | 92.2  |
| BERT (Devlin et al., 2018)      | 92.8  |
| (Akbik et al., 2019)            | 93.1  |
| (Baevski et al., 2019)          | 93.5  |
| LUKE (Yamada et al., 2020)      | 94.3  |
| LUKE w/o entity attention       | 94.1  |
| RoBERTa (Yamada et al., 2020)   | 92.4  |
| RoBERTa-extended*               | 92.54 |
| KBI*                            | 92.73 |
| KBIR*                           | 92.97 |

### Question Answering
```
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained("bloomberg/KBIR")
model = AutoModelForQuestionAnswering.from_pretrained("bloomberg/KBIR")

from datasets import load_dataset

dataset = load_dataset("squad")
```
Reported Results:

| Model                  | EM    | F1    |
|------------------------|-------|-------|
| BERT                   | 84.2  | 91.1  |
| XLNet                  | 89.0  | 94.5  |
| ALBERT                 | 89.3  | 94.8  |
| LUKE                   | 89.8  | 95.0  |
| LUKE w/o entity attention | 89.2  | 94.7  |
| RoBERTa                | 88.9  | 94.6  |
| RoBERTa-extended*      | 88.88 | 94.55 |
| KBI*                   | 88.97 | 94.7  |
| KBIR*                  | 89.04 | 94.75 |

## Any other classification task
As mentioned above since KBIR is built on top of the RoBERTa architecture, it is compatible with any AutoModel setting that RoBERTa is also compatible with. 

We encourage you to try fine-tuning KBIR on different datasets and report the downstream results.

## Contact
For any questions contact mkulkarni24@bloomberg.net
