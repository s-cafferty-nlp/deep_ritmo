# Deep Ritmo: A Rhythmic Search Engine with BERT

## Deep Ritmo is a tool that allows users to create a custom rhythmic search engine from a list of words. The current version is optimized for Spanish, but it can be adapted for any language.

# How it works:

## Deep Ritmo takes a list of words and creates a collection of embeddings for subwords in either xsampa format or IPA. The input words are weighted and encoded via a custom BERT model to create a rhythmic representation of the word. A LSH-based search engine is then able to extract the most similar words from the vocabulary based on a given input word.

## Check out this [Jupyter notebook](https://github.com/s-cafferty-nlp/deep_ritmo/blob/main/deep_ritmo_trainer_BERT.ipynb) to see how the BERT model was trained.

## Check out this [Jupyter notebook](https://github.com/s-cafferty-nlp/deep_ritmo/blob/main/deep_ritmo.ipynb) to see Deep Ritmo in action.

```
>>> import deep_ritmo

>>> word_list = ['some', 'long', 'list', 'of', 'words']

>>> R = deep_ritmo.Rhythmizer()

>>> R.add_word_list(word_list)

>>> R.query_all_vocab('vandalizar')

[OUTPUT]
0	escandalizar	  0.971598
1	analizados	    0.962546
2	analizada	    0.958395
3	analizaban	    0.958255
4	escandaliza	    0.957477
5	banalizar	    0.956744
6	paralizando	    0.954651
7	alcanzando	    0.954586
8	analizarán	    0.953820
9	andaluza	    0.953720
...
```
