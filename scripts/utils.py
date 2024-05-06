import torch
from transformers import GPT2LMHeadModel, GPT2Config, GPT2TokenizerFast, DataCollatorForLanguageModeling, DataCollatorWithPadding, DefaultDataCollator
from datasets import load_from_disk, disable_caching, IterableDataset, IterableDatasetDict
device = "cuda" if torch.cuda.is_available() else "cpu"
import types, collections

def init_model(context=None):
    print(f'Initializing model with context_size {context}...')
    if context == 'bigram':
        # configuration = GPT2Config(n_positions=8)
        configuration = GPT2Config()
    else:
        configuration = GPT2Config()
    model = GPT2LMHeadModel(configuration)
    print('...done\n')
    print(model.config)
    return model

def load_datasetdict(tokenized_data_dir, tokenizer, context_direction='left', disable_cache=True):
    print(f'Loading {tokenized_data_dir=}...')
    if disable_cache:
        print('...disabling cache while loading dataset...')
        disable_caching()
    tokenized_dataset_dict = load_from_disk(tokenized_data_dir)
    if context_direction != 'bidi':
        print('...done\n')
        return tokenized_dataset_dict
    else:
        bidi_tokenized_dataset_dict = make_bidi_iterabledatasetdict(tokenized_dataset_dict, tokenizer, tokenized_data_dir)
        print('...done\n')
        return bidi_tokenized_dataset_dict
    

def load_pretrained_model(pretrained_model_name_or_path):
    print(f'Loading pretrained model from {pretrained_model_name_or_path}...')
    model = GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path)
    print('...done\n')
    return model

def load_pretrained_tokenizer(pretrained_model_name_or_path, context_size=None, context_direction='left', add_prefix_space=False):
    print(f'Loading pretrained tokenizer from {pretrained_model_name_or_path}...')
    tokenizer = GPT2TokenizerFast.from_pretrained(
        pretrained_model_name_or_path, 
        add_prefix_space=add_prefix_space, #?
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

def bidi_expand(example, special_tokens_ids):
    BLANK_id, FILLER_id, SEP_id, BOS_id, EOS_id = special_tokens_ids

    input_ids = example['input_ids']
    # attention_mask = features['attention_mask']

    n_tokens = len(input_ids)
    
    for i in range(n_tokens):
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
        yield bidi_input

def gen_bidi_inputs(dataset, tokenizer):
    special_tokens = ['[BLANK]', '[FILLER]', '[SEP]', '<s>', '</s>']
    special_tokens_ids = [tokenizer.convert_tokens_to_ids(token) for token in special_tokens]
    for example in dataset:
        yield from bidi_expand(example, special_tokens_ids)

def make_bidi_iterabledataset(dataset, tokenizer, split_path):
    bidi_iterabledataset = IterableDataset.from_generator(gen_bidi_inputs, gen_kwargs={'dataset': dataset, 'tokenizer': tokenizer})
    print(dataset)
    try:
        # It's expensive to calculate the bidirectional dataset's length since it requires
        # looping through the entire dataset. So, I create a file in the split's path to save the length
        with open(split_path+'/length.txt', 'r') as f:
            length = int(f.readline())
    except:
        print(split_path)
        length = calculate_bidi_dataset_length(dataset)

    # To train by epoch, the dataset's CLASS needs to have a __len__ method.
    # We can't easily add a __len__ method to the IterableDataset class, and since we initialize 
    # the dataset with IterableDataset.from_generator and not __init__, we can't just initialize
    # our dataset as a subclass using __init__. So, I instead create a subclass with the __len__ 
    # method, then manually change the .__class__ attribute of the dataset.
    class BidiIterableDataset(IterableDataset):
        def __len__(self):
            return length
        
    bidi_iterabledataset.__class__ = BidiIterableDataset
    return bidi_iterabledataset

def calculate_bidi_dataset_length(dataset):
    print('Calculating bidi dataset length...')
    length = sum(len(example['input_ids']) for example in dataset)
    print(f'...done ({length=})\n')
    return length

def make_bidi_iterabledatasetdict(datasetdict, tokenizer, tokenized_data_dir):
    bidi_iterabledatasetdict = IterableDatasetDict({
        split: make_bidi_iterabledataset(dataset, tokenizer, tokenized_data_dir+'/'+split) for split, dataset in datasetdict.items()
    })
    return bidi_iterabledatasetdict


class BidiDataCollator(DefaultDataCollator):
    def __call__(self, examples):
        input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(e['input_ids']) for e in examples], batch_first=True, padding_value=0)
        attention_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor(e['attention_mask']) for e in examples], batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(e['labels']) for e in examples], batch_first=True, padding_value=-100)  # Assuming -100 is your ignore index

        batch = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
        return batch

def init_data_collator(tokenizer, context_direction='left'):
    print(f'Initializing data collator with {context_direction=}...')
    if context_direction == 'left':
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    elif context_direction == 'right':
        data_collator = ReverseSequenceDataCollator(tokenizer, mlm=False)
    elif context_direction == 'bidi':
        data_collator = BidiDataCollator()
    print('...done\n')
    return data_collator

if __name__ == "__main__":
    model = load_pretrained_model('gpt2')
    tokenizer = load_pretrained_tokenizer('gpt2')
    data_collator = init_data_collator(tokenizer, 'left')

