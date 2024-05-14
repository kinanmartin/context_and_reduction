# from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_from_disk
import torch
from tqdm import tqdm
from utils import *

from torch.nn import CrossEntropyLoss
import torch

from torch.utils.data import DataLoader


def calculate_perplexity(model, tokenizer, data_collator, test_dataset, batch_size=32):
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    dataloader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        collate_fn=data_collator,
    )
    
    total_loss = 0
    total_examples = 0

    with torch.no_grad():
        for batch in (pbar := tqdm(dataloader)):
            inputs = {key: val.to(model.device) for key, val in batch.items()}
            outputs = model(**inputs)
            loss = outputs.loss
            batch_loss = loss.item() * inputs["input_ids"].size(0)
            batch_size = inputs["input_ids"].size(0)
            # print(batch_loss)
            # print(batch_size)
            total_loss += batch_loss
            total_examples += batch_size
            # print(total_loss)
            # print()
            pbar.set_description(f"Perplexity: {torch.exp(torch.tensor(loss.item())):.2f}, Total loss: {total_loss:16.2f}")

    mean_loss = total_loss / total_examples
    
    # Perplexity is the exponential of the cross-entropy (mean loss)
    perplexity = torch.exp(torch.tensor(mean_loss))

    return perplexity.item()



if __name__ == "__main__":
    from datasets import disable_caching
    disable_caching()

    from argparse import ArgumentParser
    parser = ArgumentParser()
    # parser.add_argument("--input_data_dir")
    # parser.add_argument("--need_to_tokenize")
    parser.add_argument("--tokenized_data_dir")
    parser.add_argument("--model_dir")
    parser.add_argument("--per_device_batch_size", type=int, default=8)
    parser.add_argument("--context_size", default='sentence')
    parser.add_argument("--context_direction", default='left')

    args = parser.parse_args()

    model_dir = args.model_dir
    # model_dir = "models/script1/left_sentence/checkpoint-75047"
    # model_dir = "gpt2"

    model = load_pretrained_model(model_dir)
    tokenizer = load_pretrained_tokenizer(
        'gpt2', 
        context_size=args.context_size, 
        context_direction=args.context_direction,
        padding=True) # shouldn't change anything now
    model.resize_token_embeddings(len(tokenizer))

    data_collator = init_data_collator(tokenizer, args.context_direction, args.context_size)

    # tokenized_testset_dir = args.tokenized_data_dir
    # tokenized_testset_dir = "data/coca_spoken/tokens_sentence/test"

    tokenized_dataset_dict = load_datasetdict(
        args.tokenized_data_dir,
        disable_cache=True
    )

    test_set = tokenized_dataset_dict['test']

    # print(f'Loading {tokenized_testset_dir=}...')
    # test_set = load_from_disk(tokenized_testset_dir)
    # test_set = test_set.remove_columns('text')
    # print('...done')

    print(test_set[42])

    if args.context_direction != 'bidi':
        test_set = test_set.remove_columns(['word_ids'])

    test_set = test_set.remove_columns(['text'])
    print(test_set[42])


    perplexity = calculate_perplexity(model, tokenizer, data_collator, test_set,
                                      batch_size=args.per_device_batch_size)
    print(f"Perplexity: {perplexity}")

    # from coca_tokenize import load_data_in_splits
    # coca_dir = "data/coca_spoken/text_sentence_cleaned/"

    # coca_dsdict = load_data_in_splits(coca_dir, .8, .1, .1)
    # test_set = coca_dsdict['test']

    # testset_dir = "data/coca_spoken/text_sentence_cleaned"
    # print(f'Loading {testset_dir=}...')
    # test_set = load_dataset(testset_dir)
    # print('...done')

    # example_batch = test_set[0:5]
    # print(example_batch['input_ids'])  # Check the first few batches for structure
    # print(example_batch['attention_mask'])

    # perplexity = calculate_perplexity(model, tokenizer, test_set, 'cpu')
    # print(f"Perplexity: {perplexity}")


# python3 scripts/perplexity.py --tokenized_data_dir=data/coca_spoken_detokenized/tokens_sentence --model_dir=models/gpt2/left_sentence/checkpoint-76066 --per_device_batch_size=64 --context_size=sentence --context_direction=left ;
# python3 scripts/perplexity.py --tokenized_data_dir=data/coca_spoken_detokenized/tokens_sentence --model_dir=models/gpt2/right_sentence/checkpoint-76066 --per_device_batch_size=64 --context_size=sentence --context_direction=right ;
# python3 scripts/perplexity.py --tokenized_data_dir=data/coca_spoken_detokenized/tokens_sentence --model_dir=models/gpt2/bidi_sentence/checkpoint-76066 --per_device_batch_size=64 --context_size=sentence --context_direction=bidi ;
# python3 scripts/perplexity.py --tokenized_data_dir=data/coca_spoken_detokenized/tokens_bigram --model_dir=models/gpt2/left_bigram/checkpoint-63819 --per_device_batch_size=1024 --context_size=bigram --context_direction=left ;
# python3 scripts/perplexity.py --tokenized_data_dir=data/coca_spoken_detokenized/tokens_bigram --model_dir=models/gpt2/right_bigram/checkpoint-63819 --per_device_batch_size=1024 --context_size=bigram --context_direction=right ;
# python3 scripts/perplexity.py --tokenized_data_dir=data/coca_spoken_detokenized/tokens_trigram --model_dir=models/gpt2/bidi_bigram/checkpoint-100000 --per_device_batch_size=512 --context_size=bigram --context_direction=bidi ;


# 54.615543365478516
# 43.24224090576172
# 27.41640281677246
# 66.11920166015625
# 101.02252960205078
# 30.830102920532227


