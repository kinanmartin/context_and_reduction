from tqdm import tqdm
from datasets import load_dataset, DatasetDict, load_from_disk
from transformers import GPT2TokenizerFast
from utils import load_pretrained_tokenizer
from tokenizers.processors import TemplateProcessing
from pathlib import Path

def load_data_in_splits(data_dir, train=0.8, val=0.1, test=0.1, seed=41):
    print(f'Loading data from {data_dir} (split ratios {(train, val, test)})...')
    data = load_dataset(data_dir)
    train_valtest = data['train'].train_test_split(test_size = 1 - train, seed=seed)
    test_valid = train_valtest['test'].train_test_split(test_size = test / (val + test), seed=seed)
    out = DatasetDict({
            'train': train_valtest['train'],
            'val': test_valid['train'],
            'test': test_valid['test']
        })
    print('...done')
    return out

def tokenize_data(dataset_dict, tokenizer, context_size):
    print('Tokenizing dataset_dict with tokenizer...')

    if context_size == 'sentence' and '[BOS]' not in tokenizer.get_vocab():
        tokenizer.add_tokens('[BOS]')

    def tokenize_func(x):
        out = tokenizer(
            ['[BOS] ' + text for text in x['text']] if context_size == 'sentence' else x['text'],
            truncation=True,
            )
            # tokenizer postprocess!!
            # out.insert(0, tokenizer.bos_token)
            # out.append(tokenizer.eos_token)
        return out

    encoded_datasets = dataset_dict.map(
        tokenize_func, 
        batched=True)
    print('...done')
    return encoded_datasets



if __name__ == '__main__':
    from datasets import disable_caching
    disable_caching()

    from argparse import ArgumentParser
    parser = ArgumentParser()
    # parser.add_argument("--input_data_dir")
    # parser.add_argument("--need_to_tokenize")
    parser.add_argument("--text_dir", type=str, default=None)
    parser.add_argument("--tokens_dir", type=str)
    parser.add_argument("--context_size", default='sentence')
    parser.add_argument("--only_load", type=bool, default=False)

    args = parser.parse_args()

    input_dir = args.text_dir
    output_dir = args.tokens_dir
    context_size = args.context_size
    only_load = args.only_load


    # tokenizer = load_pretrained_tokenizer('gpt2')
    tokenizer = load_pretrained_tokenizer(
        'gpt2', 
        context_size=context_size,
        context_direction=None,
        # adding prefix space will mess up [BOS] token
        add_prefix_space=False if context_size == 'sentence' else True,
        padding=False,
        )

    # assert False
    # tokenized_data_path = 'data/coca_spoken_detokenized/tokens_sentence/'

    # coca_dir = "data/coca_spoken_detokenized/text_sentence/"

    if only_load:
        # Load pretokenized data:
        print('loading pretokenized data')
        encoded_datasets = load_from_disk(output_dir)
        print('done')
    else:
        coca_dsdict = load_data_in_splits(input_dir, .8, .1, .1)
        # Tokenize from data and save:
        encoded_datasets = tokenize_data(coca_dsdict, tokenizer, context_size)

        encoded_datasets.save_to_disk(output_dir)


    print(encoded_datasets)
    print(encoded_datasets['train'][0])

    tokenized_datasets = encoded_datasets.remove_columns(['text'])

    print(encoded_datasets['train'][0:11])

    # import random

    # print(coca_dsdict['train'].column_names)
    # for split in ['train', 'val', 'test']:
    #     print(split, random.choice(coca_dsdict[split]))


    # tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    # context_length = 1024