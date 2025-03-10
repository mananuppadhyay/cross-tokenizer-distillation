## Approach
This repository is an implementation of Cross Tokenizer distillation, taken from the paper cited below.
I have used "microsoft/phi-3-mini-instruct" as the teacher model and "bigscience/bloomz-560m" as the student model. This aims to demonstrate the improvements possible on very small scale models by using bigger models as supervisors in addition to the regular cross-entropy loss.

The dataset used here is QED.

We observe an increase in F1 score of the bloomz-560m from 0.55 to 0.588 by using the apprach mentioned in the paper. Refer to the paper for benchmarks on the QED dataset.

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
