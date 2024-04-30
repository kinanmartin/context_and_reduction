from typing import List, Dict
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize.punkt import PunktSentenceTokenizer

def mapping_between_whitespace_words_and_nltk_tokenizer_words(
    nltk_tokenizer_words: List[str],
    whitespace_words: List[str],
) -> List[int]:
    """
    Input:
        nltk_tokenizer_words: ["I", "do", "n't", "like", "waterturtles", "."]
        whitespace_words: ["I", "don't", "like", "waterturtles."]
    Output (List of len(nltk_tokenizer_words))
        [0, 1, 1, 2, 3, 3]
    """
    mapping = []
    current_word = ''
    i = 0
    for nltk_word in nltk_tokenizer_words:
        current_word += nltk_word
        mapping.append(i)
        if whitespace_words[i] == current_word:
            i += 1
            current_word = ''

    assert mapping[-1] == len(whitespace_words) - 1, f"mapping failed: {mapping}"
    return mapping
    

def candor_turn_to_coca_style(text: str) -> List[Dict]:
    """
    Pre-processes a CANDOR Cliffhanger turn to match the style
    of COCA. Primarily, this splits contractions and punctuation
    by whitespace into new words.

    text: "I don't like waterturtles. I don't."
    output:
        [
            {
                "words": ["I", "do", "n't", "like", "waterturtles", "."],
                "mapping": [0, 1, 1, 2, 3, 3]
            },
            
            {
                "words": ["I", "do", "n't", "."],
                "mapping": [0, 1, 1, 1]
            }
        ],

    """
    sentence_tokenizer = PunktSentenceTokenizer()
    word_tokenizer = TreebankWordTokenizer()
    
    sentences = sentence_tokenizer.tokenize(text)
    out = []
    for sent in sentences:
        words = word_tokenizer.tokenize(sent)
        sent_whitespace = sent.split(' ')
        mapping = mapping_between_whitespace_words_and_nltk_tokenizer_words(words, sent_whitespace)

        sentence_dict = {
            "words": words,
            "mapping": mapping
        }
        out.append(sentence_dict)
    
    return out

if __name__ == '__main__':
    mapping = mapping_between_whitespace_words_and_nltk_tokenizer_words(
        ["I", "do", "n't", "like", "waterturtles", "."],
        ["I", "don't", "like", "waterturtles."]
    )
    print(mapping)
    print([0, 1, 1, 2, 3, 3])

    text = "I don't like waterturtles. I don't."
    out = candor_turn_to_coca_style(text)
    print(out)