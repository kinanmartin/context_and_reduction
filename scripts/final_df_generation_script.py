from tqdm import tqdm
from pathlib import Path
import pandas as pd

from utils import *
from candor_surprisals import main, concat_surprisals_to_full_df
from candor_final_df import prepare_candor_text_dfs, choose_and_convert_text_df_to_list_based_on_context_size_and_direction

# candor_outer_path = '/om2/data/public/candor'

models = {
    ('left', 'sentence'): '../models/gpt2/left_sentence/checkpoint-76066',
    ('right', 'sentence'): '../models/gpt2/right_sentence/checkpoint-76066',
    ('bidi', 'sentence'): '../models/gpt2/bidi_sentence/checkpoint-76066',
    ('left', 'bigram'): '../models/gpt2/left_bigram/checkpoint-63819',
    ('right', 'bigram'): '../models/gpt2/right_bigram/checkpoint-63819',
    ('bidi', 'bigram'): '../models/gpt2/bidi_bigram/checkpoint-100000',
}

if __name__ == '__main__':
    # from datasets import disable_caching
    # disable_caching()

    # from argparse import ArgumentParser
    # parser = ArgumentParser()

    # args = parser.parse_args()

    # big_df is a large csv of each word in candor, labeled with control predictors
    big_df = pd.read_csv('../data/candor/candor.csv')

    string_columns = ['word', 'text', 'lemma', 'pos', 'conversation_id', 'speaker']
    for col in string_columns:
        big_df[col] = big_df[col].astype(str)

    word_counts = big_df['lemma'].value_counts()
    big_df['frequency'] = big_df['lemma'].map(word_counts)
    big_df['duration'] = big_df['word_stop'] - big_df['word_start']
    big_df['ends_with_punct'] = big_df['word'].apply(lambda x: any(x.endswith(punc) for punc in {'.', ',', '?', '!'}))

    # break big_df up by conversation id and process each conversation separately:
    convos = big_df['conversation_id'].unique()
    for idx, convo in (pbar := tqdm(enumerate(convos))):
        pbar.set_description(f"Processing convo ({idx}/{len(convos)}) (id: {convo})")

        convo_df = big_df[big_df['conversation_id'] == convo]
        text_sentence, text_bigram, text_trigram = prepare_candor_text_dfs(convo_df)
        
        all_model_surprisals = {}
        for (context_direction, context_size), model_dir in models.items():
            texts = choose_and_convert_text_df_to_list_based_on_context_size_and_direction(text_sentence, text_bigram, text_trigram, context_size, context_direction)
            model_surprisals = main(texts, model_dir, context_size, context_direction)
            all_model_surprisals[(context_direction, context_size)] = model_surprisals

        
        out_df = concat_surprisals_to_full_df(convo_df, {'_'.join(k): v for k, v in all_model_surprisals.items()})

        out_df.to_csv(f'../results/by_convo/convo_{idx}_{convo.replace('-', '_')}.csv')





