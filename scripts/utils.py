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
        bidi_tokenized_dataset_dict = make_bidi_iterabledatasetdict(tokenized_dataset_dict, tokenizer)
        print('...done\n')
        return bidi_tokenized_dataset_dict
    

def load_pretrained_model(pretrained_model_name_or_path):
    print(f'Loading pretrained model from {pretrained_model_name_or_path}...')
    model = GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path)
    print('...done\n')
    return model

def load_pretrained_tokenizer(pretrained_model_name_or_path, context=None, add_prefix_space=False):
    print(f'Loading pretrained tokenizer from {pretrained_model_name_or_path}...')
    tokenizer = GPT2TokenizerFast.from_pretrained(
        pretrained_model_name_or_path, 
        add_prefix_space=add_prefix_space, #?
    )

    if context == 'bigram':
        tokenizer.bos_token = '<s>'
        tokenizer.eos_token = '</s>'

    tokenizer.pad_token = tokenizer.eos_token # ?
    print("Vocabulary size:", tokenizer.vocab_size)
    print("Max Model Input Sizes:", tokenizer.model_max_length)
    print("BOS token:", tokenizer.bos_token, tokenizer.bos_token_id)
    print("EOS token:", tokenizer.eos_token, tokenizer.eos_token_id)
    print("PAD token:", tokenizer.pad_token, tokenizer.pad_token_id)
    print("SEP token:", tokenizer.sep_token, tokenizer.sep_token_id)
    print("UNK token:", tokenizer.unk_token, tokenizer.unk_token_id)
    print("Special tokens:", tokenizer.all_special_tokens)
    print('...done\n')
    return tokenizer

class ReverseSequenceDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, features, return_tensors=None):
        for feature in features:
            feature['input_ids'] = feature['input_ids'][::-1]
        return super().__call__(features, return_tensors)

def bidi_expand(example, tokenizer):
    BLANK = '[BLANK]'
    FILLER = '[FILLER]'
    SEP = '[SEP]'
    BOS = '<s>'
    EOS = '</s>'
    # num_added_tokens = tokenizer.add_tokens([BLANK, FILLER, SEP, BOS, EOS])
    # also resize tokenizer vocabulary!

    BLANK_id = tokenizer.convert_tokens_to_ids(BLANK)
    FILLER_id = tokenizer.convert_tokens_to_ids(FILLER)
    SEP_id = tokenizer.convert_tokens_to_ids(SEP)
    BOS_id = tokenizer.convert_tokens_to_ids(BOS)
    EOS_id = tokenizer.convert_tokens_to_ids(EOS)


    input_ids = example['input_ids']
    # attention_mask = features['attention_mask']

    n_tokens = len(input_ids)
    
    for i in range(n_tokens):
        bidi_input_ids = [BOS_id] +  input_ids[:i] + [BLANK_id] + input_ids[i+1:] + [EOS_id] + [SEP_id, FILLER_id]
        bidi_attention_mask = [1] * (n_tokens + 4)
        bidi_labels = ([-100] * (n_tokens + 3)) + [input_ids[i]] 
        
        bidi_input = {
            'input_ids': bidi_input_ids,
            'attention_mask': bidi_attention_mask,
            'labels': bidi_labels
        }
        # print(example)
        # print()
        # print(bidi_input)
        # print()
        # print()

        assert len(bidi_input_ids) == len(bidi_attention_mask) == len(bidi_labels)
        yield bidi_input

def gen_bidi_inputs(dataset, tokenizer):
    for example in dataset:
        yield from bidi_expand(example, tokenizer)

def make_bidi_iterabledataset(dataset, tokenizer):
    bidi_iterabledataset = IterableDataset.from_generator(gen_bidi_inputs, gen_kwargs={'dataset': dataset, 'tokenizer': tokenizer})
    # print(type(bidi_iterabledataset))
    length = calculate_bidi_dataset_length(dataset)
    class BidiIterableDataset(IterableDataset):
        def __len__(self):
            return length
        
    bidi_iterabledataset.__class__ = BidiIterableDataset
    # print(isinstance(bidi_iterabledataset, collections.abc.Sized))
    return bidi_iterabledataset

def calculate_bidi_dataset_length(dataset):
    print('Calculating bidi dataset length...')
    length = sum(len(example['input_ids']) for example in dataset)
    print(f'...done ({length=})\n')
    return length




# def make_bidi_iterabledatasetdict(datasetdict, tokenizer):
#     bidi_iterabledatasetdict = IterableDatasetDict({
#         split: BidiIterableDataset(dataset, tokenizer) for split, dataset in datasetdict.items()
#     })
#     print(bidi_iterabledatasetdict)
#     print(bidi_iterabledatasetdict['train'])
#     print(bidi_iterabledatasetdict['train'].__len__)
#     print(bidi_iterabledatasetdict['train'].__len__())
#     return bidi_iterabledatasetdict

def make_bidi_iterabledatasetdict(datasetdict, tokenizer):
    bidi_iterabledatasetdict = IterableDatasetDict({
        split: make_bidi_iterabledataset(dataset, tokenizer) for split, dataset in datasetdict.items()
    })
    # print(bidi_iterabledatasetdict)
    # print(bidi_iterabledatasetdict['train'])
    # print(bidi_iterabledatasetdict['train'].__len__)
    # print(bidi_iterabledatasetdict['train'].__len__())
    return bidi_iterabledatasetdict

# def make_bidi_iterabledataset(dataset, tokenizer):
#     # Create the IterableDataset from the generator
#     bidi_iterabledataset = IterableDataset.from_generator(gen_bidi_inputs, gen_kwargs={'dataset': dataset, 'tokenizer': tokenizer})
    
#     # Define a __len__ method that calculates the length based on the original dataset
#     def calculate_bidi_dataset_length(self):
#         return sum(len(example['input_ids']) for example in dataset)

#     # Attach the __len__ method to your bidi_iterabledataset
#     bidi_iterabledataset.__len__ = types.MethodType(calculate_bidi_dataset_length, bidi_iterabledataset)

#     return bidi_iterabledataset

# class BidiIterableDataset(IterableDataset):
#     def __init__(self, dataset, tokenizer):
#         super().__init__()
#         self.dataset = dataset
#         self.tokenizer = tokenizer
    
#     def __iter__(self):
#         for example in self.dataset:
#             yield from bidi_expand(example, tokenizer)
    
#     def __len__(self):
#         return sum(len(example['input_ids']) for example in self.dataset)


# class BidirectionalInfillingDataCollator(DataCollatorForLanguageModeling):
#     """
#     Modifies the DataCollatorForLanguageModeling to return
#     input_ids, labels, and attention_ids
#     as per pqian11/fragment-completion code (Qian and Levy, 2022)
#     suitable for bidirectional infilling task

#     From a single input (token) sentence, I should be able to create a whole batch
#     of bidirectional task inputs where each successive token is masked.
#     """
#     def __call__(self, features, return_tensors='pt', 
#                  BLANK_id=-2000, SEP_id=-1000, FILLER_id=-3000):
#         """
#         Given:
#             features := Dict{
#                 'input_ids': List, 
#                 'attention_mask': List
#             }
#         Returns:
#             batch := transformers.tokenization_utils_base.BatchEncoding{
#                 'input_ids': Tensor,
#                 'attention_mask': Tensor,
#                 'labels': Tensor
#             }

#         Example:
#             Given:
#                 input_ids = [1544, 7224, 1243, 845, 2092, 764]
#                 attention_mask = [1, 1, 1, 1, 1, 1]

#             Return:
#                 bidi_input_ids = [1544, BLANK, 1243, 845, 2092, 764, SEP, FILL]
#                 bidi_attention_mask = [1, 1, 1, 1, 1, 1, 1, 1]
#                 bidi_labels = [BLANK, BLANK, BLANK, BLANK, BLANK, BLANK, BLANK, 7224]

#             * Ensure automatic shifting of labels doesn't happen in the model

#             (Note: Padding not necessary if we make the batch from the single sentence
#             Otherwise, we need to additionally pad all inputs with zeros at the end)

#         """
#         assert isinstance(features, dict), f"bidirectional data collator input features should be a dict, not {type(features)}"
#         assert return_tensors == 'pt', f"only supports return pytorch tensors"

#         feature = features
#         input_ids = feature['input_ids']

#         n_tokens = len(input_ids)

#         bidi_input_ids = [input_ids[:i] + [BLANK_id] + input_ids[i+1:] + [SEP_id, FILLER_id] 
#                         for i in range(n_tokens)]

#         bidi_attention_mask = [[1 for _ in range(n_tokens + 2)] for _ in range(n_tokens)]

#         bidi_labels = [[-100 for _ in range(n_tokens + 1)] + [answer_token] 
#                     for answer_token in input_ids]

#         batch = {
#             'input_ids': torch.tensor(bidi_input_ids),
#             'attention_mask': torch.tensor(bidi_attention_mask),
#             'labels': torch.tensor(bidi_labels)
#         }
#         # print(batch)
#         return batch
#         # return super().__call__(features, return_tensors)


class CustomDataCollator(DefaultDataCollator):
    def __call__(self, examples):
        input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(e['input_ids']) for e in examples], batch_first=True, padding_value=0)
        attention_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor(e['attention_mask']) for e in examples], batch_first=True, padding_value=0)
        labels = torch.nn.utils.rnn.pad_sequence([torch.tensor(e['labels']) for e in examples], batch_first=True, padding_value=-100)  # Assuming -100 is your ignore index

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

def init_data_collator(tokenizer, context_direction='left'):
    print(f'Initializing data collator with {context_direction=}...')
    if context_direction == 'left':
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    elif context_direction == 'right':
        data_collator = ReverseSequenceDataCollator(tokenizer, mlm=False)
    elif context_direction == 'bidi':
        data_collator = CustomDataCollator()
        # data_collator = DataCollatorWithPadding(tokenizer)
        # data_collator = BidirectionalInfillingDataCollator(tokenizer, mlm=False)
    print('...done\n')
    return data_collator

if __name__ == "__main__":
    model = load_pretrained_model('gpt2')
    tokenizer = load_pretrained_tokenizer('gpt2')
    data_collator = init_data_collator(tokenizer, 'left')

