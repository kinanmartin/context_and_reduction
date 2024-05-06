import torch
from transformers import GPT2LMHeadModel, GPT2Config, GPT2TokenizerFast, DataCollatorForLanguageModeling, DataCollatorWithPadding, DefaultDataCollator, PreTrainedTokenizerFast
from datasets import load_from_disk, disable_caching, IterableDataset, IterableDatasetDict
device = "cuda" if torch.cuda.is_available() else "cpu"
import types, collections
import random

def init_model(context_size=None):
    print(f'Initializing model with context_size {context_size}...')
    # if context_size == 'bigram':
    #     # configuration = GPT2Config(n_positions=8)
    configuration = GPT2Config()
    model = GPT2LMHeadModel(configuration)
    print('...done\n')
    print(model.config)
    return model

def load_datasetdict(tokenized_data_dir, tokenizer, disable_cache=True):
    print(f'Loading {tokenized_data_dir=}...')
    if disable_cache:
        print('...disabling cache while loading dataset...')
        disable_caching()
    tokenized_dataset_dict = load_from_disk(tokenized_data_dir)
    print('...done\n')
    return tokenized_dataset_dict
    

def load_pretrained_model(pretrained_model_name_or_path):
    print(f'Loading pretrained model from {pretrained_model_name_or_path}...')
    model = GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path)
    print('...done\n')
    return model

def load_pretrained_tokenizer(pretrained_model_name_or_path, context_size=None, context_direction='left', add_prefix_space=False):
    print(f'Loading pretrained tokenizer from {pretrained_model_name_or_path}...')
    tokenizer = GPT2TokenizerFast.from_pretrained(
        pretrained_model_name_or_path, 
        add_prefix_space=add_prefix_space, # AssertionError: You need to instantiate GPT2TokenizerFast with add_prefix_space=True to use it with pretokenized inputs.
    )

    # if context_size == 'bigram':
    tokenizer.bos_token = '<s>'
    tokenizer.eos_token = '</s>'

    if context_direction == 'bidi':
        special_tokens = ['[BLANK]', '[FILLER]', '[SEP]',]
            # 'BOS': '<s>',
            # 'EOS': '</s>',
        tokenizer.add_tokens(special_tokens)

    tokenizer.pad_token = tokenizer.eos_token # ?
    print("Vocabulary size:", tokenizer.vocab_size)
    print("Max Model Input Sizes:", tokenizer.model_max_length)
    print("Special tokens:", tokenizer.all_special_tokens)
    print('...done\n')
    return tokenizer

class ReverseSequenceDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, features, return_tensors=None):
        for feature in features:
            feature['input_ids'] = feature['input_ids'][::-1]
        return super().__call__(features, return_tensors)
    

class BidiDataCollator(DataCollatorWithPadding):

    def _save_special_tokens_ids(self):
        special_tokens = ['[BLANK]', '[FILLER]', '[SEP]', '<s>', '</s>']
        self.special_tokens_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in special_tokens]

    def __call__(self, features, return_tensors=None):
        try:
            self.special_tokens_ids
        except AttributeError:
            self._save_special_tokens_ids()

        bidi_features = [make_bidi_input(feature, self.special_tokens_ids, seed=1029) for feature in features]
        print(bidi_features)
        return super().__call__(bidi_features, return_tensors)
    
def make_bidi_input(feature, special_tokens_ids, seed=1029):
    BLANK_id, FILLER_id, SEP_id, BOS_id, EOS_id = special_tokens_ids

    input_ids = feature['input_ids']
    # attention_mask = features['attention_mask']

    n_tokens = len(input_ids)

    random.seed(seed)
    i = random.randint(0, len(input_ids)-1)
    
    # bidi_input_ids = [BOS_id] +  input_ids[:i] + [BLANK_id] + input_ids[i+1:] + [EOS_id] + [SEP_id, FILLER_id]
    # bidi_attention_mask = [1] * (n_tokens + 4)
    # bidi_labels = ([-100] * (n_tokens + 3)) + [input_ids[i]] 

    bidi_input_ids = input_ids[:i] + [BLANK_id] + input_ids[i+1:] + [SEP_id, FILLER_id]
    bidi_attention_mask = [1] * (n_tokens + 2)
    bidi_labels = ([-100] * (n_tokens + 1)) + [input_ids[i]] 
    
    bidi_input = {
        'input_ids': bidi_input_ids,
        'attention_mask': bidi_attention_mask,
        'labels': bidi_labels
    }
    assert len(bidi_input_ids) == len(bidi_attention_mask) == len(bidi_labels)
    return bidi_input


def init_data_collator(tokenizer, context_direction='left'):
    print(f'Initializing data collator with {context_direction=}...')
    if context_direction == 'left':
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    elif context_direction == 'right':
        data_collator = ReverseSequenceDataCollator(tokenizer, mlm=False)
    elif context_direction == 'bidi':
        data_collator = BidiDataCollator(tokenizer)
    print('...done\n')
    return data_collator

if __name__ == "__main__":
    model = load_pretrained_model('gpt2')
    tokenizer = load_pretrained_tokenizer('gpt2')
    model.resize_token_embeddings(len(tokenizer))
    data_collator = init_data_collator(tokenizer, 'left')

