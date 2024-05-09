from tqdm import tqdm
from datasets import load_dataset, DatasetDict, load_from_disk
from transformers import GPT2TokenizerFast
from utils import load_pretrained_tokenizer

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

def tokenize_data(dataset_dict, tokenizer, context_size, batch_size=1000):
    print('Tokenizing dataset_dict with tokenizer...')

    # def tokenize_func(x):
    #     out = tokenizer(
    #         ['[BOS] ' + text + ' [EOS]' for text in x['text']] if context_size == 'sentence' else x['text'],
    #         truncation=True,
    #         )
    #     return out
    
    def tokenize_func(examples):
        """
        Splits text on whitespace before tokenizing, uses is_split_into_words=True.
        datasets.map will treat the BatchEncoding output as a dict, and so we will lose the word/token mapping information.
        Here, we save the Fast tokenizer word/token mappings by adding them as another key:value in out before returning
        """
        if context_size == 'sentence':
            split_text = [['[BOS]'] + text.split(' ') + ['[EOS]'] for text in examples['text']]
        else:
            split_text = [text.split(' ') for text in examples['text']]

        out = tokenizer(
            split_text,
            truncation=True,
            is_split_into_words=True,
            )

        out['word_ids'] = [out.word_ids(i) for i in range(len(out['input_ids']))]
        # # Sanity check:
        # for i in range(3):
        #     print(examples['text'][i])
        #     print(out['input_ids'][i])
        #     print(out['word_ids'][i])
        # assert False
        return out

    encoded_datasets = dataset_dict.map(
        tokenize_func, 
        batched=True,
        batch_size=batch_size,
        writer_batch_size=batch_size)
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
    parser.add_argument("--only_load", type=str, default='n')
    parser.add_argument("--batch_size", type=int, default=1000)

    args = parser.parse_args()

    input_dir = args.text_dir
    output_dir = args.tokens_dir
    context_size = args.context_size
    only_load = args.only_load
    batch_size = args.batch_size


    # tokenizer = load_pretrained_tokenizer('gpt2')
    tokenizer = load_pretrained_tokenizer(
        'gpt2', 
        context_size=context_size,
        context_direction=None,
        # adding prefix space will mess up [BOS] token
        # add_prefix_space=False if context_size == 'sentence' else True,
        add_prefix_space=True,
        padding=False,
        )

    # assert False
    # tokenized_data_path = 'data/coca_spoken_detokenized/tokens_sentence/'

    # coca_dir = "data/coca_spoken_detokenized/text_sentence/"

    if only_load == 'y':
        # Load pretokenized data:
        print('loading pretokenized data')
        encoded_datasets = load_from_disk(output_dir)
        print('done')
    else:
        coca_dsdict = load_data_in_splits(input_dir, .8, .1, .1)
        # Tokenize from data and save:
        encoded_datasets = tokenize_data(coca_dsdict, tokenizer, context_size, batch_size)

        encoded_datasets.save_to_disk(output_dir)


    print(encoded_datasets)
    print(encoded_datasets['train'][0])

    tokenized_datasets = encoded_datasets.remove_columns(['text'])

    print(encoded_datasets['train'][0:11])
    print(type(encoded_datasets['train'][0]))

    print(tokenizer(encoded_datasets['train'][0]['text'], truncation=True))
    print(type(tokenizer(encoded_datasets['train'][0]['text'], truncation=True)))
    print()
    # print(encoded_datasets['train'][0].words())

    # import random

    # print(coca_dsdict['train'].column_names)
    # for split in ['train', 'val', 'test']:
    #     print(split, random.choice(coca_dsdict[split]))


    # tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    # context_length = 1024