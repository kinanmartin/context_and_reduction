from pathlib import Path
import pandas as pd

from candor.create_raw_data import load_conversation_tokens

from candor_control_predictors import ControlPredictors
from surprisals import *
from utils import *

def load_transcript_output(convo_path: Path):
    # convo_path = Path(convo_path)
    output_df = load_conversation_tokens(convo_path)
    output_df = output_df[output_df.type == 'pronunciation']    
    return output_df

def load_transcribe_cliffhanger(convo_path: Path):
    # convo_path = Path(convo_path)
    return pd.read_csv(convo_path / 'transcription/transcript_cliffhanger.csv')

def candor_full_df(cliffhanger_df, output_df):
    """
    Given transcribe_output.json and transcript_cliffhanger.csv
    as loaded dataframes, return new cliffhanger_exploded df 
    mapping each word and their start/stop times to cliffhanger_df
    """
    row_starts = []
    row_stops = []
    row_words = []
    # row_surprisals = []
    row_control_predictors = {}

    for idx, row in cliffhanger_df.iterrows():
        output_sub_df = output_df.query(
            "start >= @row.start & stop <= @row.stop & speaker == @row.speaker"
        )

        row_starts.append(output_sub_df.start.tolist())
        row_stops.append(output_sub_df.stop.tolist())

        output_words = output_sub_df.utterance.tolist()

        cliffhanger_words = row['utterance'].split(' ')#.strip?
        row_words.append(cliffhanger_words) # or output_words, to remove punctuation

        control_predictors = ControlPredictors(row['utterance'])
        for predictor, values in control_predictors:
            if predictor not in row_control_predictors:
                row_control_predictors[predictor] = [values]
            else:
                row_control_predictors[predictor].append(values)
    
        # # Surprisals:
        # inputs = tokenize_cliffhanger_turn(cliffhanger_words, tokenizer)
        # surprisals = calculate_surprisal(inputs, model)
        # surprisals_by_word = aggregate_surprisal_by_word(inputs, surprisals)

        assert len(output_words) == len(cliffhanger_words), f"output/cliffhanger transcript mismatch:\n{output_words}\n{cliffhanger_words}\n"
        # assert len(cliffhanger_words) == len(surprisals_by_word), f"cliffhanger_words/surprisals_by_word mismatch:\n{cliffhanger_words}\n{surprisals_by_word}\n"

    cliffhanger_df_minimal = cliffhanger_df.loc[:, ['turn_id']]
    cliffhanger_df_minimal["word"] = cliffhanger_df["utterance"].str.split(' ')
    cliffhanger_df_minimal["word_start"] = row_starts
    cliffhanger_df_minimal["word_stop"] = row_stops
    # cliffhanger_df_minimal["surprisal"] = row_surprisals
    for predictor, values in row_control_predictors.items():
        cliffhanger_df_minimal[predictor] = values


    full_df = cliffhanger_df_minimal.explode(["word", "word_start", "word_stop",] + [predictor for predictor in row_control_predictors])# "surprisal"])
    full_df["position_in_turn"] = full_df.groupby("turn_id").cumcount()
    return full_df.reset_index(drop=True)

def make_df_from_convo_path(convo_path, model, tokenizer, out_path=None, save_type='pickle'):
    # convo_path = Path(convo_path)
    # out_path = Path(out_path)
    cliffhanger_df = load_transcribe_cliffhanger(convo_path)
    output_df = load_transcript_output(convo_path)
    cliffhanger_exploded = candor_full_df(cliffhanger_df, output_df, model, tokenizer)
    if out_path is not None:
        if save_type == 'pickle':
            cliffhanger_exploded.to_pickle(out_path / (convo_path.name + '.pickle'))
        elif save_type == 'csv':
            cliffhanger_exploded.to_csv(out_path / (convo_path.name + '.csv'))
    return cliffhanger_exploded

def prepare_candor_text_dfs(full_df):
    all_sentences = []
    all_bigrams = []
    all_trigrams = []

    # Iterate through each group of sentences
    for (turn_id, sentence_id), group in full_df.groupby(['turn_id', 'sentence_id_in_turn']):
        words = list(group['word'])
        all_sentences.append({
            'turn_id': turn_id, 
            'sentence_id_in_turn': sentence_id,
            'sentence': ' '.join(words)})

        for bigram in gen_bigrams(words):
            all_bigrams.append({
            'turn_id': turn_id, 
            'bigram': bigram})

        for trigram in gen_trigrams(words):
            all_trigrams.append({
            'turn_id': turn_id, 
            'trigram': trigram})

    text_sentence = pd.DataFrame(all_sentences)
    text_bigram = pd.DataFrame(all_bigrams)
    text_trigram = pd.DataFrame(all_trigrams)
    return {
        'text_sentence': text_sentence,
        'text_bigram': text_bigram,
        'text_trigram': text_trigram,
    }

def compute_candor_surprisals(model, tokenizer, cliffhanger_df, input_df, context_size, context_direction):
    def tokenize_func(tokenizer, text):
        if context_size == 'sentence':
            split_text = ['[BOS]'] + text.split(' ') + ['[EOS]']
        else:
            split_text = text.split(' ')

        out = tokenizer(
            split_text,
            truncation=True,
            is_split_into_words=True,
            )

        out['word_ids'] = out.word_ids()
        # # Sanity check:
        # for i in range(3):
        #     print(examples['text'][i])
        #     print(out['input_ids'][i])
        #     print(out['word_ids'][i])
        # assert False
        return out

    all_surprisals = []
    
    for turn_id, group in input_df.groupby('turn_id'):
        turn_surprisals = []
        for text in group[context_size]:
            input = tokenize_func(tokenizer, text)
            surprisals = calculate_per_word_surprisals(input, model, context_size, context_direction)
            turn_surprisals.extend(surprisals)
        all_surprisals.append(turn_surprisals)
    
    return all_surprisals

def concat_surprisals_to_full_df(full_df, all_surprisals):

    return 

if __name__ == '__main__':
    # convo_path = Path('data/candor/sample/0020a0c5-1658-4747-99c1-2839e736b481/')
    # out_path = Path('data/candor/exploded/')

    from argparse import ArgumentParser
    parser = ArgumentParser()
    # parser.add_argument("--input_data_dir")
    # parser.add_argument("--need_to_tokenize")
    parser.add_argument("--candor_convo_path", type=Path)
    parser.add_argument("--model_dir", type=Path)
    parser.add_argument("--out_path", type=Path, default=None)
    parser.add_argument("--per_device_batch_size", type=int, default=8)
    parser.add_argument("--context_size", default='sentence')
    parser.add_argument("--context_direction", default='left')

    args = parser.parse_args()

    model_dir = args.model_dir
    # model_dir = "models/script1/left_sentence/checkpoint-75047"
    # model_dir = "gpt2"
    tokenizer_name = "gpt2"

    model = load_pretrained_model(model_dir)
    tokenizer = load_pretrained_tokenizer(
        tokenizer_name, 
        context_size=args.context_size, 
        context_direction=args.context_direction, 
        add_prefix_space=True
    )
    model.resize_token_embeddings(len(tokenizer))

    candor_df = make_df_from_convo_path(args.candor_convo_path, model, tokenizer, args.out_path, save_type='csv')
    