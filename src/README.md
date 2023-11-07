# Usage

## Lexical Semantic Difference Detection
To detect words whose meanings are different in the input two corpora (source and target corpora).
```bash
python detect_meaning_differences.py SOURCE_CORPUS TARGET_CORPUS
```
The input corpora, SOURCE_CORPUS and TARGET_CORPUS, are text files with the sentence-per-line format. Upper/lower cases are ignored by default, but they are considered by the options (--cased and also use a corresponding model --bert_model).

The output format is: Word TAB Score TAB Frequency of word in source corpus TAB Frequency of word in target corpus. The lines are sorted in descending order according to the score, meaning that the higher the rank is, the larger difference the word has in the source and target corpora.


## Representative Word Instance Extraction
To extract word instances of a specified word type having wider meanings in the source corpus than in the target corpus. are different in the input two corpora (source and target corpora).
```bash
python find_representative_word_instances.py -t TARGET_PHRASE SOURCE_CORPUS TARGET_CORPUS
```
The input corpora are again text files with the sentence-per-line format. The target phrase can be explored by the semantic difference detection program above. The program accepts a word (e.g., -t 'better') and also a phrase (e.g., -t 'better off') as the target phrase.  

The output format is: Score TAB context before the target phrase TAB target phrase TAB context after. The lines are sorted in descending order according to the score, meaning that the higher the rank is, the more unique the word instance is in the source corpus; conversely, the lower, the more common to both source and target corpora.
