import random

import torch
import nltk
import numpy as np

from seem_ros.utils.constants import IMAGENET_DEFAULT_TEMPLATES

nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

def get_tag(tokenized, tags):
    if not isinstance(tags, (list, tuple)):
        tags = [tags]
    ret = []
    for (word, pos) in nltk.pos_tag(tokenized):
        for tag in tags:
            if pos == tag:
                ret.append(word)
    return ret

def get_noun_phrase(tokenized):
    # Taken from Su Nam Kim Paper...
    grammar = r"""
        NBAR:
            {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns

        NP:
            {<NBAR>}
            {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
    """
    chunker = nltk.RegexpParser(grammar)

    chunked = chunker.parse(nltk.pos_tag(tokenized))
    continuous_chunk = []
    current_chunk = []

    for subtree in chunked:
        if isinstance(subtree, nltk.Tree):
            current_chunk.append(' '.join([token for token, pos in subtree.leaves()]))
        elif current_chunk:
            named_entity = ' '.join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue

    return continuous_chunk

def text_noun_with_prompt_all(text, phrase_prob=0.0, append_text=True):
    tokenized = nltk.word_tokenize(text)
    
    if random.random() >= phrase_prob:
        nouns = get_tag(tokenized, ['NN', 'NNS', 'NNP'])
    else:
        nouns = get_noun_phrase(tokenized)


    prompt_texts = [np.random.choice(IMAGENET_DEFAULT_TEMPLATES).format(noun) for noun in nouns]
    
    if append_text:
        prompt_texts += [text]
        nouns += [text]
    
    return prompt_texts, nouns