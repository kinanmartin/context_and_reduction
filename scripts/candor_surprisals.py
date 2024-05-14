import torch
from datasets import Dataset
from torch.utils.data import DataLoader

from coca_tokenize import tokenize_data
from tqdm import tqdm

from utils import *
from candor_final_df import *
from typing import Dict, List
# from datasets import Features, Value


from datasets import disable_caching
disable_caching()


def tokenize_candor_text(text, tokenizer, context_size):

    if context_size == 'sentence':
        split_text = ['[BOS]'] + text.split(' ') + ['[EOS]']
    else:
        split_text = text.split(' ')

    out = tokenizer(
        split_text,
        truncation=True,
        is_split_into_words=True,
        # return_tensors='pt',
        )
    return out


def calculate_surprisal(inputs, model, context_size, context_direction):
    """
    Given a tokenized encoding `inputs` of type 
        transformers.tokenization_utils_base.BatchEncoding

    Return a tensor of surprisals for each token.
    """
    # print(inputs)
    input_ids = inputs['input_ids']
    if context_direction == 'right':
        input_ids = input_ids[::-1]
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    input_ids = torch.tensor([input_ids])
    with torch.no_grad():
        input_ids = input_ids.to(device)
        outputs = model(input_ids, labels=input_ids)
        logit_predictions = outputs.logits
        
        shift_logits = logit_predictions[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        surprisals = loss.view(shift_labels.size()).cpu().numpy().tolist()
        
    surprisals = [None] + surprisals[0] # dummy value for first item so length is same as input length
    if context_direction == 'right':
        surprisals.reverse()
    return surprisals

def aggregate_sentence_token_surprisals(inputs, token_surprisals):
    """
    Aggregate (add) surprisals of tokens which come
    from the same word.
    """
    n_words = inputs.word_ids()[-1] + 1
    # print(inputs.word_ids())
    word_surprisals = []
    for word_idx in range(1, n_words - 1): # range skips [BOS] and [EOS]
        tokenspan = inputs.word_to_tokens(word_idx)
        # print(tokenspan)
        token_start, token_end = tokenspan.start, tokenspan.end
        # print(token_surprisals[token_start: token_end])
        word_surprisal = sum(token_surprisals[i] for i in range(token_start, token_end))
        word_surprisals.append(word_surprisal)
    # print(token_surprisals)
    # print(word_surprisals)
    return word_surprisals

def aggregate_bigram_token_surprisals(inputs, token_surprisals, context_direction):
    """
    Aggregate (add) surprisals of tokens which come
    from the same word.
    """
    if context_direction in ['left', 'bidi']:
        word_idx = 1
    else:
        word_idx = 0
    tokenspan = inputs.word_to_tokens(word_idx)
    # print(tokenspan)
    token_start, token_end = tokenspan.start, tokenspan.end
    # print(token_surprisals[token_start: token_end])
    word_surprisal = sum(token_surprisals[i] for i in range(token_start, token_end))
    # print(token_surprisals)
    # print(word_surprisals)
    return word_surprisal

def gen_all_bidi_inputs(inputs, special_tokens_ids):
    BLANK_id, FILLER_id, SEP_id = special_tokens_ids

    input_ids = inputs['input_ids']
    n_tokens = len(input_ids)
    n_words = inputs.word_ids()[-1] + 1

    for word_id_to_mask in range(1, n_words - 1): # range skips [BOS] and [EOS]
        tokenspan = inputs.word_to_tokens(word_id_to_mask)
        # print(tokenspan)
        token_mask_start_idx, token_mask_end_idx = tokenspan.start, tokenspan.end

        mask_len = token_mask_end_idx - token_mask_start_idx + 1

        bidi_input_ids = input_ids[:token_mask_start_idx] + [BLANK_id] + input_ids[token_mask_end_idx+1:] + [SEP_id] + ([FILLER_id] * mask_len)
        bidi_attention_mask = [1] * (n_tokens + 2)
        bidi_labels = ([-100] * (n_tokens + 2 - mask_len)) + input_ids[token_mask_start_idx:token_mask_end_idx+1]
        
        bidi_input = {
            'input_ids': bidi_input_ids,
            'attention_mask': bidi_attention_mask,
            'labels': bidi_labels
        }

        assert len(bidi_input_ids) == len(bidi_attention_mask) == len(bidi_labels), f"lengths don't match: {bidi_input_ids}\n{bidi_attention_mask}\n{bidi_labels}"
        yield bidi_input

def calculate_bidi_sentence_surprisal(inputs, model):
    # print(inputs)
    with torch.no_grad():
        inputs = {key: torch.tensor([val]).to(model.device) for key, val in inputs.items()}
        outputs = model(**inputs)
        loss = outputs.loss

    return loss.item()


def compute_candor_surprisals(model, tokenizer, 
                              texts, 
                              context_size, context_direction):

    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    if context_size == 'sentence' and context_direction == 'bidi':
        special_tokens_ids = [tokenizer.convert_tokens_to_ids(token) for token in ['[BLANK]', '[FILLER]', '[SEP]']]

    all_surprisals = []

    if context_size == 'bigram':
        memo = {}
    for text in (pbar := tqdm(texts)):
        if context_size == 'bigram' and text in memo:
            all_surprisals.append(memo[text])
            continue

        inputs = tokenize_candor_text(text, tokenizer, context_size)
        # print(text)
        if context_size == 'sentence' and context_direction == 'bidi':
            word_surprisals = []
            for bidi_sentence_inputs in gen_all_bidi_inputs(inputs, special_tokens_ids):
                surprisal = calculate_bidi_sentence_surprisal(bidi_sentence_inputs, model)
                word_surprisals.append(surprisal)

        else:
            token_surprisals = calculate_surprisal(inputs, model, context_size, context_direction)
            # print(token_surprisals)
            if context_size == 'sentence':
                word_surprisals = aggregate_sentence_token_surprisals(inputs, token_surprisals)
            else:
                # print(inputs)
                # print(inputs.word_ids())
                # print(token_surprisals)
                word_surprisals = aggregate_bigram_token_surprisals(inputs, token_surprisals, context_direction)
            # print(word_surprisals)
                memo[text] = word_surprisals
        
        all_surprisals.append(word_surprisals)

    assert len(all_surprisals) == len(texts)
            
    return all_surprisals

def unpack_sentence_surprisals(all_surprisals: List[List[float]]) -> List[float]:
    return [surprisal for sentence in all_surprisals for surprisal in sentence]

def concat_surprisals_to_full_df(full_df: pd.DataFrame, 
                                 all_surprisals_all_models: Dict[str, List[float]]):
    out_df = full_df.copy()
    for context_size_and_dir, all_surprisals in all_surprisals_all_models.items():
        assert len(all_surprisals) == len(full_df)
    
        out_df[context_size_and_dir] = all_surprisals

    return out_df


def main(texts: List[str], model_dir, context_size, context_direction):
    # candor_convo_path = Path(candor_convo_path)

    model = load_pretrained_model(model_dir)
    tokenizer = load_pretrained_tokenizer(
        'gpt2', 
        context_size=context_size, 
        context_direction=context_direction, 
        add_prefix_space=True
    )
    model.resize_token_embeddings(len(tokenizer))

    # full_df = make_df_from_convo_path(candor_convo_path)#, model, tokenizer)#, args.out_path, save_type='csv')
    
    # text_sentence, text_bigram, text_trigram = prepare_candor_text_dfs(full_df)

    all_surprisals = compute_candor_surprisals(
        model, tokenizer,
        texts, context_size, context_direction,
    )

    # for text, word_surprisals in zip(texts, all_surprisals):
    #     print(text, word_surprisals)
    if context_size == 'sentence':
        all_surprisals = unpack_sentence_surprisals(all_surprisals)
    return all_surprisals


if __name__ == '__main__':
    from pathlib import Path


    from datasets import disable_caching
    disable_caching()

    from argparse import ArgumentParser
    parser = ArgumentParser()
    # parser.add_argument("--input_data_dir")
    # parser.add_argument("--need_to_tokenize")
    parser.add_argument("--candor_convo_path")
    parser.add_argument("--model_dir")
    # parser.add_argument("--per_device_batch_size", type=int, default=8)
    parser.add_argument("--context_size", default='sentence')
    parser.add_argument("--context_direction", default='left')

    args = parser.parse_args()

    # model_dir = 'gpt2'
    # model_dir = 'models/gpt2/left_sentence/checkpoint-76066'
    model_dir = args.model_dir

    context_size = args.context_size
    # context_direction = 'left'
    context_direction = args.context_direction
    
    # candor_convo_path = Path('data/candor/sample/0020a0c5-1658-4747-99c1-2839e736b481/')
    candor_convo_path = args.candor_convo_path

    main(candor_convo_path, model_dir, context_size, context_direction)

    # model = load_pretrained_model(model_dir)
    # tokenizer = load_pretrained_tokenizer(
    #     tokenizer_name, 
    #     context_size=context_size, 
    #     context_direction=context_direction, 
    #     add_prefix_space=True
    # )
    # model.resize_token_embeddings(len(tokenizer))
    # data_collator = init_data_collator(tokenizer, context_direction, context_size)

    # full_df = make_df_from_convo_path(candor_convo_path)#, model, tokenizer)#, args.out_path, save_type='csv')
    
    # text_sentence, text_bigram, text_trigram = prepare_candor_text_dfs(full_df)

    # # tokenized_sentence_dataset = tokenize_candor_text_df(text_sentence, tokenizer,
    # #                                                      context_size)

    # # print(text_sentence['text'].tolist())
    # # tokenized_sentence_encodings = tokenize_candor_texts(
    # #     text_sentence['text'].tolist(),
    # #     tokenizer,
    # #     context_size
    # # )

    # # print(tokenized_sentence_encodings[0])

    # if context_size == 'sentence':
    #     texts = text_sentence
    # elif context_size == 'bigram':
    #     if context_direction == 'left':
    #         texts = text_bigram[~text_bigram.text.str.contains('</s>')]
    #     elif context_direction == 'right':
    #         texts = text_bigram[~text_bigram.text.str.contains('<s>')]
    #     else: # 'bidi'
    #         texts = text_trigram

    # texts = texts['text'].tolist()

    # all_surprisals = compute_candor_surprisals(
    #     model, tokenizer,
    #     texts, context_size, context_direction,
    # )

    # for text, word_surprisals in zip(texts, all_surprisals):
    #     print(text, word_surprisals)

    
    
