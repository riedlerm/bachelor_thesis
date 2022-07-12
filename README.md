Transfer Learning for Multilingual and Monolingual Sentiment Analysis
======

This project contains the source code for my bachelor thesis "Transfer Learning for Multilingual and Monolingual Sentiment Analysis".

For the thesis a series of experiments was conducted to examine the performance of cross-lingual transfer learning in both multilingual and monolingual settings and to inspect how well transfer learning can perform across languages.
For this purpose, three experimental settings as well as a baseline have been implemented by fine-tuning different BERT models on multi-class sentiment analysis on Amazon reviews. The languages taken into consideration are English, German, Chinese and Japanese and the dataset used for all experiments is the Multilingual Amazon Reviews Corpus (MARC) (https://aclanthology.org/2020.emnlp-main.369.pdf).
In the baseline approach a monolingual BERT model is fine-tuned on each of the four languages considered and then directly evaluated on each language in a zero-shot transfer setting. The experimental approaches include a multilingual setting, where multilingual BERT models are transferred to a target language by zero-shot transfer, a monolingual setting, that transfers knowledge from a source language to a target language by learning new token embeddings, and a feature-based approach, where BERT embeddings are extracted and used as fixed features for traditional machine learning algorithms.


1 Baseline
------

- Fine-tuning of a language specific BERT model for each of the 4 languages considered (Englsich, German, Chinese, Japanese)
- Zero-shot evaluation of each model on the 4 languages

2 Multilingual Transfer
------

Includes three settings where a multilingual BERT is fine-tuned on different language combinations to inspect the impact of the number of languages used for fine-tuning and to examine how well a model can perform based on the selected language combinations:

- Multi_ALL
- Multi_PAIR
- Multi_SINGLE

### Multi_ALL:

Multilingual BERT model simultaneously fine-tuned on English, German, Japanese and Chinese and then evaluated on each of the four languages.

### Multi_PAIR:

Multilingual BERT model fine-tuned on two of the four languages simultaneously. The languages are selected in a way to obtain both pairs of languages within the same language family and from different families:

- English - German
- English - Chinese
- Japanese - German
- Japanese - Chinese

### Multi_SINGLE:

Multilingual BERT fine-tuned on each of the four languages individually.

3 Monolingual Transfer
------

The models trained in the monolingual transfer setting are a simplified implementation of the approach proposed by Artetxe et al. (2020) (https://aclanthology.org/2020.acl-main.421.pdf). For the experiments carried out in this thesis, English is used as source language and German as target language.
The following simplifications were adopted:
- Instead of performing a dedicated step for model pre-training, as is done in the paper, the pre-trained English BERT from Hugging Face is used, due to the high computational costs of pre-training a model from scratch.
- The token embeddings in the target language are either extracted from a pre-trained model or learned during fine-tuning instead of during pre-training

4 Feature-Based Transfer
------

In this approach, instead of focusing on the prediction of the fine-tuned model, the goal is to extract the embeddings of Transformer layer 11 of the BERT architecture. The obtained embeddings were then used as fixed features for three traditional machine learning algorithms for text classification:
- Naive-Bayes classifier
- Stochastic gradient descent
- Decision tree

References
------

Artetxe, M., Ruder, S., and Yogatama, D. (2020). On the cross-lingual transferability of monolingual representations. _In Proceedings of the 58th Annual Meeting of the
Association for Computational Linguistics_, pages 4623–4637, Online. Association for Computational Linguistics.
(https://aclanthology.org/2020.acl-main.421.pdf)

Keung, P., Lu, Y., Szarvas, G., and Smith, N. A. (2020). The multilingual Amazon reviews
corpus. _In Proceedings of the 2020 Conference on Empirical Methods in Natural Lan-
guage Processing (EMNLP)_, pages 4563–4568, Online. Association for Computational
Linguistics. (https://aclanthology.org/2020.emnlp-main.369.pdf)
