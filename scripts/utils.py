import torch
from transformers import GPT2LMHeadModel, GPT2Config, GPT2TokenizerFast, DataCollatorForLanguageModeling, DataCollatorWithPadding, DefaultDataCollator, PreTrainedTokenizerFast
from datasets import load_from_disk, disable_caching, IterableDataset, IterableDatasetDict
device = "cuda" if torch.cuda.is_available() else "cpu"
import types, collections
import random
from bisect import bisect_left, bisect_right

def init_model(context_size=None):
    print(f'Initializing model with context_size {context_size}...')
    # if context_size == 'bigram':
    #     # configuration = GPT2Config(n_positions=8)
    configuration = GPT2Config()
    model = GPT2LMHeadModel(configuration)
    print('...done\n')
    print(model.config)
    return model

def load_datasetdict(tokenized_data_dir, disable_cache=True):
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

def load_pretrained_tokenizer(pretrained_model_name_or_path, 
                              context_size='sentence', 
                              context_direction='left', 
                              add_prefix_space=True,
                              padding=False):
    print(f'Loading pretrained tokenizer from {pretrained_model_name_or_path}...')
    tokenizer = GPT2TokenizerFast.from_pretrained(
        pretrained_model_name_or_path, 
        add_prefix_space=add_prefix_space, # AssertionError: You need to instantiate GPT2TokenizerFast with add_prefix_space=True to use it with pretokenized inputs.
        padding=padding
    )

    if context_size == 'sentence':
        tokenizer.add_special_tokens({'bos_token': '[BOS]',
                                      'eos_token': '[EOS]'})
    if context_size == 'bigram':
        tokenizer.add_special_tokens({'bos_token': '<s>',
                                      'eos_token': '</s>',})

    if context_direction == 'bidi':
        tokenizer.add_special_tokens({
            'mask_token': '[BLANK]',
            'cls_token': '[FILLER]',
            'sep_token': '[SEP]'
            })

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
    

class BidiDataCollator(DefaultDataCollator):
    random.seed(1299)

    def __init__(self, tokenizer, context_size, special_tokens=['[BLANK]', '[FILLER]', '[SEP]']):
        self.tokenizer = tokenizer
        self.trigram = context_size == 'trigram'
        self.special_tokens = special_tokens
        self.special_tokens_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in special_tokens]

    def __call__(self, features):
        bidi_features = [make_bidi_input(feature, self.special_tokens_ids, self.trigram) for feature in features]

        input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(e['input_ids']) for e in bidi_features], batch_first=True, padding_value=0)
        attention_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor(e['attention_mask']) for e in bidi_features], batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(e['labels']) for e in bidi_features], batch_first=True, padding_value=-100)  # Assuming -100 is your ignore index

        batch = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
        return batch


def find_first_last_indices(arr, target):
    first_index = bisect_left(arr, target)
    last_index = bisect_right(arr, target) - 1
    return first_index, last_index

def make_bidi_input(feature, special_tokens_ids, trigram=False):
    BLANK_id, FILLER_id, SEP_id = special_tokens_ids

    input_ids = feature['input_ids']
    # attention_mask = features['attention_mask']
    word_ids = feature['word_ids']

    n_tokens = len(input_ids)

    if trigram:
        # example:
        # words: [<s>, I'm, not]
        # word_ids: [0, 1, 1, 2]  # length is len(tokens), word_ids[-1] is len(words)
        # in this case, we must mask word_id=1
        assert word_ids[-1] == 2, f'tokenized trigram has too many words: {word_ids}'
        word_id_to_mask = 1
    else:
        word_id_to_mask = random.randint(1, word_ids[-1]-1) # 1 and -1 to exclude [BOS] and [EOS]
    
    token_mask_start_idx, token_mask_end_idx = find_first_last_indices(word_ids, word_id_to_mask)
    mask_len = token_mask_end_idx - token_mask_start_idx + 1

    bidi_input_ids = input_ids[:token_mask_start_idx] + [BLANK_id] + input_ids[token_mask_end_idx+1:] + [SEP_id] + ([FILLER_id] * mask_len)
    bidi_attention_mask = [1] * (n_tokens + 2)
    bidi_labels = ([-100] * (n_tokens + 1)) + input_ids[token_mask_start_idx:token_mask_end_idx+1]
    
    bidi_input = {
        'input_ids': bidi_input_ids,
        'attention_mask': bidi_attention_mask,
        'labels': bidi_labels
    }
    assert len(bidi_input_ids) == len(bidi_attention_mask) == len(bidi_labels)
    return bidi_input


def init_data_collator(tokenizer, context_direction='left', context_size=None):
    print(f'Initializing data collator with {context_direction=}...')
    if context_direction == 'left':
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    elif context_direction == 'right':
        data_collator = ReverseSequenceDataCollator(tokenizer, mlm=False)
    elif context_direction == 'bidi':
        data_collator = BidiDataCollator(tokenizer, context_size)
    print('...done\n')
    return data_collator

if __name__ == "__main__":
    model = load_pretrained_model('gpt2')
    tokenizer = load_pretrained_tokenizer('gpt2')
    model.resize_token_embeddings(len(tokenizer))
    data_collator = init_data_collator(tokenizer, 'left')

