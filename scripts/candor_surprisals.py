import torch
from datasets import Dataset
from torch.utils.data import DataLoader

from coca_tokenize import tokenize_data
from tqdm import tqdm

from utils import *
from candor_final_df import *
# from datasets import Features, Value

# def tokenize_candor_text_df(text_df, tokenizer, context_size, batch_size=1000):
#     text_df = text_df.drop(columns=text_df.columns.difference(['text']))
#     text_df_dataset = Dataset.from_pandas(text_df)

#     # print(text_df_dataset)
#     tokenized_text_df_dataset = tokenize_data(text_df_dataset, tokenizer, context_size, batch_size)
#     # print(tokenized_text_df_dataset)
#     # print(tokenized_text_df_dataset[42])
#     return tokenized_text_df_dataset
#     # if context_size == 'bidi':
#     #     return tokenized_text_df_dataset.remove_columns(['text'])
#     # return tokenized_text_df_dataset.remove_columns(['text', 'word_ids'])

def tokenize_candor_text(text, tokenizer, context_size):

    if context_size == 'sentence':
        split_text = ['[BOS]'] + text.split(' ') + ['[EOS]']
    else:
        split_text = text.split(' ')

    out = tokenizer(
        split_text,
        truncation=True,
        is_split_into_words=True,
        return_tensors='pt',
        )
    return out

# def aggregate_sentence_token_surprisals(inputs, batch_surprisals):
#     """
#     [BOS] token's surprisal is already not calculated.
#     [EOS] token's surprisal is dropped here.
#     """

#     sentences = []
#     for batch_idx, token_surprisals in enumerate(batch_surprisals):
#         token_word_mapping = inputs['word_ids'][batch_idx]
#         n_words = token_word_mapping[-1] + 1
#         word_level_surprisals = [0] * n_words
#         for token_idx, word_idx in enumerate(token_word_mapping):
#             word_level_surprisals[word_idx] += token_surprisals[token_idx]
#         sentences.append(word_level_surprisals)
#     return sentences

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
    
    with torch.no_grad():
        input_ids = input_ids.to(device)
        outputs = model(input_ids, labels=input_ids)
        logit_predictions = outputs.logits
        
        shift_logits = logit_predictions[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        surprisals = loss.view(shift_labels.size()).cpu().numpy().tolist()
        
    surprisals = [None] + surprisals[0]
    if context_direction == 'right':
        surprisals.reverse()
    return surprisals

def aggregate_surprisal_by_word(inputs, token_surprisals):
    """
    Aggregate (add) surprisals of tokens which come
    from the same word.
    """
    n_words = inputs.word_ids()[-1]
    # print(inputs.word_ids())
    word_surprisals = []
    for word_idx in range(1, n_words):
        tokenspan = inputs.word_to_tokens(word_idx)
        # print(tokenspan)
        token_start, token_end = tokenspan.start, tokenspan.end
        # print(token_surprisals[token_start-1: token_end-1])
        # token_surprisals is missing the 0 index token which corresponds to [BOS], so we index into i-1
        word_surprisal = sum(token_surprisals[i] for i in range(token_start, token_end))
        word_surprisals.append(word_surprisal)
    # print(token_surprisals)
    # print(word_surprisals)
    return word_surprisals


def compute_candor_surprisals(model, tokenizer, 
                              texts, 
                              context_size, context_direction):

    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    all_surprisals = []

    for text in (pbar := tqdm(texts)):
        inputs = tokenize_candor_text(text, tokenizer, context_size)
        # print(text)
        token_surprisals = calculate_surprisal(inputs, model, context_size, context_direction)
        # print(token_surprisals)
        word_surprisals = aggregate_surprisal_by_word(inputs, token_surprisals)
        # print(word_surprisals)
        all_surprisals.append(word_surprisals)

    assert len(all_surprisals) == len(texts)
            
    return all_surprisals

# def compute_candor_surprisals(model, tokenizer, data_collator, 
#                               tokenized_text_df_dataset, 
#                               context_size, context_direction, 
#                               batch_size=64):

#     model.eval()

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model.to(device)

#     dataloader = DataLoader(
#         tokenized_text_df_dataset, 
#         batch_size=batch_size, 
#         collate_fn=data_collator,

#     )

#     loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

#     all_surprisals = []
    
#     with torch.no_grad():
#         for batch in (pbar := tqdm(dataloader)):
#             inputs = {key: val.to(model.device) for key, val in batch.items()}
#             outputs = model(**inputs)

#             if context_direction in ['left', 'right']:
#                 if context_size == 'sentence':
#                     logit_predictions = outputs.logits
                    
#                     shift_logits = logit_predictions[..., :-1, :].contiguous()
#                     shift_labels = inputs['input_ids'][..., 1:].contiguous()
                    
#                     loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                    
#                     batch_surprisals = loss.view(shift_labels.size()).cpu().numpy().tolist()
#                     # print(batch_surprisals)
#                     batch_word_surprisals = aggregate_sentence_token_surprisals(inputs, batch_surprisals)
                    
#                 elif context_size == 'bigram':
#                     NotImplementedError


#             elif context_direction == 'bidi':
#                 if context_size == 'sentence':
#                     NotImplementedError
                
#                 elif context_size == 'bigram':
#                     NotImplementedError


#             all_surprisals.extend(batch_word_surprisals)

#     assert len(all_surprisals) == len(tokenized_text_df_dataset)
            
#     # for turn_id, group in input_df.groupby('turn_id'):
#     #     turn_surprisals = []
#     #     for text in group[context_size]:
#     #         input = tokenize_func(tokenizer, text)
#     #         surprisals = calculate_per_word_surprisals(input, model, context_size, context_direction)
#     #         turn_surprisals.extend(surprisals)
#     #     all_surprisals.append(turn_surprisals)
    
#     return all_surprisals

def concat_surprisals_to_full_df(full_df, all_surprisals):

    return 


if __name__ == '__main__':
    from pathlib import Path

    model_dir = 'gpt2'
    tokenizer_name = 'gpt2'

    context_size = 'sentence'
    context_direction = 'left'
    
    candor_convo_path = Path('data/candor/sample/0020a0c5-1658-4747-99c1-2839e736b481/')

    model = load_pretrained_model(model_dir)
    tokenizer = load_pretrained_tokenizer(
        tokenizer_name, 
        context_size=context_size, 
        context_direction=context_direction, 
        add_prefix_space=True
    )
    model.resize_token_embeddings(len(tokenizer))
    data_collator = init_data_collator(tokenizer, context_direction, context_size)

    full_df = make_df_from_convo_path(candor_convo_path)#, model, tokenizer)#, args.out_path, save_type='csv')
    
    text_sentence, text_bigram, text_trigram = prepare_candor_text_dfs(full_df)

    # tokenized_sentence_dataset = tokenize_candor_text_df(text_sentence, tokenizer,
    #                                                      context_size)

    # print(text_sentence['text'].tolist())
    # tokenized_sentence_encodings = tokenize_candor_texts(
    #     text_sentence['text'].tolist(),
    #     tokenizer,
    #     context_size
    # )

    # print(tokenized_sentence_encodings[0])
    texts = text_sentence['text'].tolist()

    all_surprisals = compute_candor_surprisals(
        model, tokenizer,
        texts, context_size, context_direction,
    )

    for text, word_surprisals in zip(texts, all_surprisals):
        print(text, word_surprisals)

    
    
