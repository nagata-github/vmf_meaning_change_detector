# Meaning difference detector based on the variance of word meanings.
Implementations for the methods for detecting meaning changes (differences) with their representative instances. All source codes are available under src/. The details of the methods are available in the following paper.

## Citation
Please cite the following paper if you use the above resources.  
```
@inproceedings{nagata-etal-2023-variance,
    title = "Variance Matters: Detecting Semantic Differences without Corpus/Word Alignment",
    author = "Nagata, Ryo  and Takamura, Hiroya and Otani, Naoki  and Kawasaki, Yoshifumi",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.965/",
    doi = "10.18653/v1/2023.emnlp-main.965",
    pages = "15609--15622"
 }
```
## Errata
In Subsection 3.1 in the paper above, max(Kt, Ks)/min(Kt, Ks) is described as a meaning change score. This is incorrect. Correctly, the reciprocal, min(Kt, Ks)/max(Kt, Ks) is used in the following experiment (the smaller the value, the larger the difference). Also, In 3.2, the mean of 1 - cos and 1 - min(Kt, Ks)/max(Kt, Ks) is used as a score for the Mean direction & coverage method. These are for the SemEval2020 semantic change detection experiment, and do not affect the implementations on this page.

## Contact
Ryo Nagata (nagata-emnlp2023 [at] ml.hyogo-u.ac.jp).
