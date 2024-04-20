from pathlib import Path
import pandas as pd

from candor.create_raw_data import load_conversation_tokens

def load_transcript_output(convo_path: Path):
    output_df = load_conversation_tokens(convo_path)
    output_df = output_df[output_df.type == 'pronunciation']    
    return output_df

def load_transcribe_cliffhanger(convo_path: Path):
    return pd.read_csv(convo_path / 'transcription/transcript_cliffhanger.csv')

def explode_cliffhanger(cliffhanger_df, output_df):
    """
    Given transcribe_output.json and transcript_cliffhanger.csv
    as loaded dataframes, return new cliffhanger_exploded df 
    mapping each word and their start/stop times to cliffhanger_df
    """
    row_starts = []
    row_stops = []
    row_words = []

    for idx, row in cliffhanger_df.iterrows():
        output_sub_df = output_df.query(
            "start >= @row.start & stop <= @row.stop & speaker == @row.speaker"
        )

        row_starts.append(output_sub_df.start.tolist())
        row_stops.append(output_sub_df.stop.tolist())

        output_words = output_sub_df.utterance.tolist()

        cliffhanger_words = row['utterance'].split(' ')#.strip?
        row_words.append(cliffhanger_words) # or output_words, to remove punctuation

        # print(len(output_words), len(cliffhanger_words))
        # print(output_words)
        # print(cliffhanger_words)
        assert len(output_words) == len(cliffhanger_words), f"output/cliffhanger transcript mismatch:\n{output_words}\n{cliffhanger_words}\n"


    cliffhanger_df["word_start"] = row_starts
    cliffhanger_df["word_stop"] = row_stops
    cliffhanger_df["utterance_exploded"] = cliffhanger_df["utterance"].str.split(' ')

    cliffhanger_exploded = cliffhanger_df.explode(["utterance_exploded", "word_start", "word_stop"])
    cliffhanger_exploded["position_in_turn"] = cliffhanger_exploded.groupby("turn_id").cumcount()
    return cliffhanger_exploded

if __name__ == '__main__':
    convo_path = Path('../data/candor/sample/0020a0c5-1658-4747-99c1-2839e736b481/')
    cliffhanger_df = load_transcribe_cliffhanger(convo_path)
    output_df = load_transcript_output(convo_path)
    cliffhanger_exploded = explode_cliffhanger(cliffhanger_df, output_df)
    out_path = Path('../data/candor/exploded/')
    cliffhanger_exploded.to_csv(out_path / 'sample.pkl')