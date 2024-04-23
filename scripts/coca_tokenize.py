from tqdm import tqdm
from datasets import load_dataset, DatasetDict, load_from_disk
from transformers import GPT2TokenizerFast
from utils import load_pretrained_tokenizer

def load_data_in_splits(data_dir, train=0.8, val=0.1, test=0.1):
    data = load_dataset(data_dir)
    train_valtest = data['train'].train_test_split(test_size = 1 - train)
    test_valid = train_valtest['test'].train_test_split(test_size = test / (val + test))
    out = DatasetDict({
            'train': train_valtest['train'],
            'val': test_valid['train'],
            'test': test_valid['test']
        })
    return out

def tokenize_data(dataset_dict, tokenizer):
    encoded_datasets = dataset_dict.map(
        lambda x: tokenizer(
            x['text'],
            truncation=True,
            # max_length=context_length,
            # return_overflowing_tokens=True,
            # return_length=True,
            ), 
        batched=True)
    return encoded_datasets



if __name__ == '__main__':

    coca_dir = "data/coca_spoken/text_bigram_cleaned/"

    coca_dsdict = load_data_in_splits(coca_dir, .8, .1, .1)

    tokenizer = load_pretrained_tokenizer('gpt2', context='bigram')

    print("Vocabulary size:", tokenizer.vocab_size)
    print("Max Model Input Sizes:", tokenizer.model_max_length)
    print("BOS token:", tokenizer.bos_token, tokenizer.bos_token_id)
    print("EOS token:", tokenizer.eos_token, tokenizer.eos_token_id)
    print("PAD token:", tokenizer.pad_token, tokenizer.pad_token_id)
    print("SEP token:", tokenizer.sep_token, tokenizer.sep_token_id)
    print("UNK token:", tokenizer.unk_token, tokenizer.unk_token_id)
    print("Special tokens:", tokenizer.all_special_tokens)

    # assert False
    tokenized_data_path = 'data/coca_spoken/tokens_bigram/'

    retokenize = True
    if retokenize:
        # Tokenize from data and save:
        encoded_datasets = tokenize_data(coca_dsdict, tokenizer)

        encoded_datasets.save_to_disk(tokenized_data_path)

    else:
        # Load pretokenized data:
        encoded_datasets = load_from_disk(tokenized_data_path)

    print(encoded_datasets)
    print(encoded_datasets['train'][0])

    tokenized_datasets = encoded_datasets.remove_columns(['text'])

    # import random

    # print(coca_dsdict['train'].column_names)
    # for split in ['train', 'val', 'test']:
    #     print(split, random.choice(coca_dsdict[split]))


    # tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    # context_length = 1024