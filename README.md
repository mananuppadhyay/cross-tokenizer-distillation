# Cross Tokenizer Distillation

This repository contains an implementation of **Cross Tokenizer Distillation**, inspired by the approach described in the paper cited below. The goal is to improve small-scale models by leveraging larger models as supervisors, in addition to the standard cross-entropy loss.

## Models Used
- **Teacher Model**: [`microsoft/phi-3-mini-instruct`](https://huggingface.co/microsoft/phi-3-mini-instruct)
- **Student Model**: [`bigscience/bloomz-560m`](https://huggingface.co/bigscience/bloomz-560m)

## Dataset
This implementation is trained on the **QED dataset** ([QED on Hugging Face](https://huggingface.co/datasets/qed)).

## Results
We observe an improvement in the **F1 score** of `bloomz-560m` from **0.55** to **0.588** using Cross Tokenizer Distillation. For benchmarks on the QED dataset, please refer to the original paper.


## Citations
@misc{boizard2024crosstokenizer,
      title={Towards Cross-Tokenizer Distillation: the Universal Logit Distillation Loss for LLMs}, 
      author={Nicolas Boizard and Kevin El Haddad and Céline Hudelot and Pierre Colombo},
      year={2024},
      eprint={2402.12030},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}

@misc{lamm2020qed,
    title={QED: A Framework and Dataset for Explanations in Question Answering},
    author={Matthew Lamm and Jennimaria Palomaki and Chris Alberti and Daniel Andor and Eunsol Choi and Livio Baldini Soares and Michael Collins},
    year={2020},
    eprint={2009.06354},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
