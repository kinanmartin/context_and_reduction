import torch
from transformers import GPT2LMHeadModel, GPT2Config, GPT2TokenizerFast, DataCollatorForLanguageModeling
device = "cuda" if torch.cuda.is_available() else "cpu"

def init_model(context=None):
    print(f'Initializing model with context_size {context}...')
    if context == 'bigram':
        # configuration = GPT2Config(n_positions=8)
        configuration = GPT2Config()
    else:
        configuration = GPT2Config()
    model = GPT2LMHeadModel(configuration)
    print('...done')
    print(model.config)
    return model

def load_pretrained_model(pretrained_model_name_or_path):
    print(f'Loading pretrained model from {pretrained_model_name_or_path}...')
    model = GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path)
    print('...done')
    return model

def load_pretrained_tokenizer(pretrained_model_name_or_path, context=None):
    print(f'Loading pretrained tokenizer from {pretrained_model_name_or_path}...')
    tokenizer = GPT2TokenizerFast.from_pretrained(
        pretrained_model_name_or_path, 
        # add_prefix_space=True, # ?
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
    print('...done')
    return tokenizer

class ReverseSequenceDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, features, return_tensors=None):
        for feature in features:
            feature['input_ids'] = feature['input_ids'][::-1]
        return super().__call__(features, return_tensors)

class BidirectionalInfillingDataCollator(DataCollatorForLanguageModeling):
    """
    Modifies the DataCollatorForLanguageModeling to return
    input_ids, labels, and attention_ids
    as per pqian11/fragment-completion code (Qian and Levy, 2022)
    suitable for bidirectional infilling task

    From a single input (token) sentence, I should be able to create a whole batch
    of bidirectional task inputs where each successive token is masked.
    """
    def __call__(self, features, return_tensors='pt', 
                 BLANK_id=-2000, SEP_id=-1000, FILLER_id=-3000):
        """
        Given:
            features := Dict{
                'input_ids': List, 
                'attention_mask': List
            }
        Returns:
            batch := transformers.tokenization_utils_base.BatchEncoding{
                'input_ids': Tensor,
                'attention_mask': Tensor,
                'labels': Tensor
            }

        Example:
            Given:
                input_ids = [1544, 7224, 1243, 845, 2092, 764]
                attention_mask = [1, 1, 1, 1, 1, 1]

            Return:
                bidi_input_ids = [1544, BLANK, 1243, 845, 2092, 764, SEP, FILL]
                bidi_attention_mask = [1, 1, 1, 1, 1, 1, 1, 1]
                bidi_labels = [BLANK, BLANK, BLANK, BLANK, BLANK, BLANK, BLANK, 7224]

            * Ensure automatic shifting of labels doesn't happen in the model

            (Note: Padding not necessary if we make the batch from the single sentence
            Otherwise, we need to additionally pad all inputs with zeros at the end)

        """
        assert isinstance(features, dict), f"bidirectional data collator input features should be a dict, not {type(features)}"
        assert return_tensors == 'pt', f"only supports return pytorch tensors"

        feature = features
        input_ids = feature['input_ids']

        n_tokens = len(input_ids)

        bidi_input_ids = [input_ids[:i] + [BLANK_id] + input_ids[i+1:] + [SEP_id, FILLER_id] 
                        for i in range(n_tokens)]

        bidi_attention_mask = [[1 for _ in range(n_tokens + 2)] for _ in range(n_tokens)]

        bidi_labels = [[-100 for _ in range(n_tokens + 1)] + [answer_token] 
                    for answer_token in input_ids]

        batch = {
            'input_ids': torch.tensor(bidi_input_ids),
            'attention_mask': torch.tensor(bidi_attention_mask),
            'labels': torch.tensor(bidi_labels)
        }
        # print(batch)
        return batch
        # return super().__call__(features, return_tensors)

def init_data_collator(tokenizer, context_direction='left'):
    print(f'Initializing data collator with {context_direction=}...')
    if context_direction == 'left':
        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    elif context_direction == 'right':
        data_collator = ReverseSequenceDataCollator(tokenizer, mlm=False)
    elif context_direction == 'bidi':
        data_collator = BidirectionalInfillingDataCollator(tokenizer, mlm=False)
    print('...done')
    return data_collator

if __name__ == "__main__":
    model = load_pretrained_model('gpt2')
    tokenizer = load_pretrained_tokenizer('gpt2')
    data_collator = init_data_collator(tokenizer, 'left')

