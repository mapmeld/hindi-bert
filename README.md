# hindi-bert

This is a first attempt at a Hindi language model trained with Google Research's [ELECTRA](https://github.com/google-research/electra).  **I don't modify ELECTRA until we get into finetuning**, and only then because there's hardcoded train and test files

The corpus is Hindi text (9 GB of OSCAR / CommonCrawl, ~1GB of Hindi Wikipedia)

Notebooks show finetuning classifiers on review sentiment analysis (3500 x 3 categories), BBC topic classification, and XNLI

Blog post: <a href="https://medium.com/@mapmeld/teaching-hindi-to-electra-b11084baab81">https://medium.com/@mapmeld/teaching-hindi-to-electra-b11084baab81</a>

It's available on HuggingFace: https://huggingface.co/monsoon-nlp/hindi-bert
- sample usage in HindiMovieReviews-HF.ipynb

I was greatly influenced by: https://huggingface.co/blog/how-to-train

## Corpus

Download: https://drive.google.com/drive/u/1/folders/1WikYHHMI72hjZoCQkLPr45LDV8zm9P7p

The corpus is two files:
- Hindi CommonCrawl deduped by OSCAR https://traces1.inria.fr/oscar/
- latest Hindi Wikipedia ( https://dumps.wikimedia.org/hiwiki/20200420/ ) + WikiExtractor to txt 

Bonus notes:
- Adding English wiki text or parallel corpus could help with cross-lingual tasks and training

## Vocabulary

https://drive.google.com/file/d/1-02Um-8ogD4vjn4t-wD2EwCE-GtBjnzh/view?usp=sharing

Bonus notes:
- Created with HuggingFace Tokenizers; could be longer or shorter, review ELECTRA vocab_size param

## Pretrain TF Records

[build_pretraining_dataset.py](https://github.com/google-research/electra/blob/master/build_pretraining_dataset.py) splits the corpus into training documents

Set the ELECTRA model size and whether to split the corpus by newlines.  This process can take hours on its own.

https://drive.google.com/drive/u/1/folders/1--wBjSH59HSFOVkYi4X-z5bigLnD32R5

Bonus notes:
- I am not sure of the meaning of the corpus newline split (what is the alternative?) and given this corpus, which creates the better training docs

## Training

Structure your files, with data-dir named "trainer" here

```
trainer
- vocab.txt
- pretrain_tfrecords
-- (all .tfrecord... files)
- models
-- modelname
--- checkpoint
--- graph.pbtxt
--- model.*
```

CoLab notebook gives examples of GPU vs. TPU setup

[configure_pretraining.py](https://github.com/google-research/electra/blob/master/configure_pretraining.py)

Baby Model: https://drive.google.com/drive/folders/1KPJ_rhji7Q_4qazLOMhiiG21kCFADpfS?usp=sharing

Baby2 Model (more training) https://drive.google.com/drive/folders/1cwQlWryLE4nlke4OixXA7NK8hzlmUR0c?usp=sharing

## Using the model with transformers

It's available on HuggingFace: https://huggingface.co/monsoon-nlp/hindi-bert - sample usage: https://colab.research.google.com/drive/1mSeeSfVSOT7e-dVhPlmSsQRvpn6xC05w

## Finetuning

Each task (such as XLNI, BBC, Hindi Movie Reviews) is a hardcoded class.

Where to place your training and test/dev data in the file system (for data-dir = trainer)

```
trainer
- finetuning_data
-- xnli
--- train.tsv
--- dev.tsv
- models
-- model_name
--- finetuning_tfrecords
--- finetuning_models
```

^^ If things go bad or you redesign your data, delete finetuning_tfrecords and finetuning_models

In finetune/task_builder.py

```python
elif task_name == "bbc":
    return classification_tasks.BBC(config, tokenizer)
```

In finetune/classification/classification_tasks.py

```python
class BBC(ClassificationTask):
  def __init__(self, config: configure_finetuning.FinetuningConfig, tokenizer):
    super(BBC, self).__init__(config, "bbc", tokenizer,
                               ['southasia', 'international', 'learningenglish', 'institutional', 'india', 'news', 'pakistan', 'multimedia', 'social', 'china', 'entertainment', 'science', 'business', 'sport'])

  def get_examples(self, split):
    return self._create_examples(read_tsv(
        os.path.join(self.config.raw_data_dir(self.name), split + ".csv"),
        quotechar="\"",
        max_lines=100 if self.config.debug else None), split)

  def _create_examples(self, lines, split):
    return self._load_glue(lines, split, 1, None, 0, skip_first_line=True)
```
