from pathlib import Path
import pandas as pd

from candor.create_raw_data import load_conversation_tokens

from surprisals import *
from utils import *

def load_transcript_output(convo_path: Path):
    output_df = load_conversation_tokens(convo_path)
    output_df = output_df[output_df.type == 'pronunciation']    
    return output_df

def load_transcribe_cliffhanger(convo_path: Path):
    return pd.read_csv(convo_path / 'transcription/transcript_cliffhanger.csv')

def candor_durations_and_surprisals(cliffhanger_df, output_df, model, tokenizer):
    """
    Given transcribe_output.json and transcript_cliffhanger.csv
    as loaded dataframes, return new cliffhanger_exploded df 
    mapping each word and their start/stop times to cliffhanger_df
    """
    row_starts = []
    row_stops = []
    row_words = []
    row_surprisals = []

    for idx, row in cliffhanger_df.iterrows():
        output_sub_df = output_df.query(
            "start >= @row.start & stop <= @row.stop & speaker == @row.speaker"
        )

        row_starts.append(output_sub_df.start.tolist())
        row_stops.append(output_sub_df.stop.tolist())

        output_words = output_sub_df.utterance.tolist()

        cliffhanger_words = row['utterance'].split(' ')#.strip?
        # print(cliffhanger_words)

        row_words.append(cliffhanger_words) # or output_words, to remove punctuation

        inputs = tokenize_cliffhanger_turn(cliffhanger_words, tokenizer)
        surprisals = calculate_surprisal(inputs, model)
        surprisals_by_word = aggregate_surprisal_by_word(inputs, surprisals)

        row_surprisals.append(surprisals_by_word)

        # print(len(output_words), len(cliffhanger_words))
        # print(output_words)
        # print(cliffhanger_words)
        assert len(output_words) == len(cliffhanger_words), f"output/cliffhanger transcript mismatch:\n{output_words}\n{cliffhanger_words}\n"
        assert len(cliffhanger_words) == len(surprisals_by_word), f"cliffhanger_words/surprisals_by_word mismatch:\n{cliffhanger_words}\n{surprisals_by_word}\n"

    cliffhanger_df_minimal = cliffhanger_df.loc[:, ['turn_id']]
    cliffhanger_df_minimal["word"] = cliffhanger_df["utterance"].str.split(' ')
    cliffhanger_df_minimal["word_start"] = row_starts
    cliffhanger_df_minimal["word_stop"] = row_stops
    cliffhanger_df_minimal["surprisal"] = row_surprisals


    out = cliffhanger_df_minimal.explode(["word", "word_start", "word_stop", "surprisal"])
    out["position_in_turn"] = out.groupby("turn_id").cumcount()
    return out.reset_index(drop=True)

def make_df_from_convo_path(convo_path, out_path=None, save_type='pickle'):
    cliffhanger_df = load_transcribe_cliffhanger(convo_path)
    output_df = load_transcript_output(convo_path)
    cliffhanger_exploded = candor_durations_and_surprisals(cliffhanger_df, output_df)
    if out_path is not None:
        if save_type == 'pickle':
            cliffhanger_exploded.to_pickle(out_path / (convo_path.name + '.pickle'))
        elif save_type == 'csv':
            cliffhanger_exploded.to_csv(out_path / (convo_path.name + '.csv'))
    return cliffhanger_exploded

if __name__ == '__main__':
    convo_path = Path('data/candor/sample/0020a0c5-1658-4747-99c1-2839e736b481/')
    out_path = Path('data/candor/exploded/')
    candor_df = make_df_from_convo_path(convo_path, out_path, save_type='csv')
    