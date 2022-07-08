# Description
This repository contains the experimental code used in pre-training the KBIR and KeyBART models as described in Learning Rich Representation for Keyphrases (https://arxiv.org/pdf/2112.08547.pdf) and to appear in Findings of NAACL 2022.

Some of the code builds on top of code from HuggingFace Transformers (https://github.com/huggingface/transformers) and also takes inspiration from SpanBERT (https://github.com/facebookresearch/SpanBERT)

# Running the pre-training
Use the two bash scripts for running pre-training for KBIR and KeyBART respectively.

# Accessing Pre-trained models
Models are uploaded to HuggingFace along with Model Cards describing usage.

KBIR: https://huggingface.co/bloomberg/KBIR

KeyBART: https://huggingface.co/bloomberg/KeyBART

## Citation
```
  @article{kulkarni2021kbirkeybart,
        title={Learning Rich Representation of Keyphrases from Text},
        author={Mayank Kulkarni and Debanjan Mahata and Ravneet Arora and Rajarshi Bhowmik},
        journal={arXiv preprint arXiv:2112.08547},
        year={2021}
      }
```

## License
KBIR and KeyBART are Apache 2.0. The license applies to the pre-trained models as well.

# Contact
For any questions reach out to mkulkarni24@bloomberg.net
