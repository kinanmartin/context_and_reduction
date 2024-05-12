import spacy
from nltk.corpus import cmudict

nlp = spacy.load("en_core_web_sm")
cmudict_dict = cmudict.dict()

class ControlPredictors:
    nlp = nlp
    cmudict_dict = cmudict_dict
    def __init__(self, turn):
        """
        Given a string corresponding to a turn, return dictionary of list of control predictor
        values for each WHITESPACE word of the original turn. (whitespace is important so we can
        map back to the durations that are annotated in CANDOR)
        """
        doc = nlp(turn)
        self.doc = doc
        self.turn = turn

        self.text = []
        self.lemma = []
        self.n_chars = []
        self.n_syllables = []
        self.pos = []
        self.stopword = []
        self.mtw = [] # is word a multi-token word? ex: "I've", but not "he," (punct)
        
        self.sentence_id_in_turn = []
        self.word_id_in_sentence = []

        self.n_whitespace_words = 0
        skip_this_token = False
        sentence_id = -1
        word_id = 0
        for idx, token in enumerate(doc):
            if skip_this_token:
                if token.whitespace_:
                    skip_this_token = False
                continue

            # print(token)
            
            self.text.append(token.text)
            self.lemma.append(token.lemma_)
            self.n_chars.append(len(token.text))
            self.n_syllables.append(self.word_syllable_count(token.lemma_.lower()))
            self.pos.append(token.pos_)
            self.stopword.append(token.is_stop)

            self.n_whitespace_words += 1

            if token.is_sent_start:
                sentence_id += 1
                word_id = 0

            self.sentence_id_in_turn.append(sentence_id)
            self.word_id_in_sentence.append(word_id)

            word_id += 1
            
            is_mtw = False
            if not token.whitespace_:
                skip_this_token = True
                if idx + 1 < len(doc) - 1:
                    # the current token is a multitoken word if there's no whitespace between
                    # this token and the next, and the next token is not punctuation
                    is_mtw = (doc[idx+1].pos_ != 'PUNCT') #and doc[idx+1].text != '-'

            self.mtw.append(is_mtw)
        
        assert len(turn.split(' ')) == self.n_whitespace_words, f"mismatch in spacy tokenization, actual whitespace: {len(turn.split(' '))} != spacy whitespace: {self.n_whitespace_words}"

    def word_syllable_count(self, word) -> int:
        if word in self.cmudict_dict:
            # Return the minimum count if multiple pronunciations exist
            return min([len([y for y in x if y[-1].isdigit()]) for x in self.cmudict_dict[word]])
        else:
            return None
        
    def __iter__(self):
        for k, v in vars(self).items():
            if isinstance(v, list):
                yield k, v