# Create the Heldner-Edlund transcripts from raw AWS tokens. Collate
# individual conversation transcripts into CSVs with all transcripts
# for each turn model (Heldner-Edlund, Audiophile/AWS, Backbiter, and
# Cliffhanger). Then create Heldner-Edlund turn exchange data from any
# transcript. Turn exchanges, or intervals, are described as Gaps,
# Overlaps, Pauses, and Within-Speaker Overlaps.
#
# This turn exchange algorithm was originally described in Heldner and
# Edlund (2010) but we adapt it to transcripts other than it originally
# intended.
#
# The only line that might need to be changed by the user is the
# definition of DATA_DOWNLOAD_PATH. Direct this to wherever
# the raw data are stored.
# Do not change OUTPUT_FN unless you're aware of the dependencies
# on this filename.

import json

import pandas as pd
import numpy as np

from enum import Enum

from pathlib import Path

from tqdm.auto import tqdm
from pqdm.processes import pqdm

from typing import Dict

from sentence_transformers import SentenceTransformer

# constants in order of most likely to need to be changed by the user

# for the multiprocessed functions, how many processes to open?
# for a 2, 4, or 8 core machine, we recommend N_JOBS set to 1,
# 3, and 6, respectively.
# on an 8 core Mac Pro with N_JOBS=6, this file runs in about 12 minutes
N_JOBS = 3

DATA_PATH = Path("data/")

# where is the data download directory relative to the current directory?
DATA_DOWNLOAD_PATH = DATA_PATH / "no_media/"

# what should the saved transcript and turn exchange CSVs be
# prefixed with?
# this naming has downstream dependencies, so we do not recommend
# changing it unless you're aware of its effect
TRANSCRIPT_OUTPUT_PREFIX = DATA_PATH / "transcripts--"
TURN_EXCHANGE_OUTPUT_PREFIX = DATA_PATH / "raw-intervals--"

# this number is defined in Heldner and Edlund (2010)
# do not change this to replicate our results
MAXIMUM_PAUSE = 20 / 1000


class Model(Enum):
    HELDNEREDLUND = "heldner-edlund"
    AUDIOPHILE = "audiophile"
    BACKBITER = "backbiter"
    CLIFFHANGER = "cliffhanger"


# the original AWS data is measured at a minimum granularity of 10ms, so
# to convert this to other, more sensible units, we define the granularity
class Granularity(Enum):
    SECOND = 1
    DECISECOND = 10
    CENTISECOND = 100
    MILLISECOND = 1000


GRANULARITY = Granularity.CENTISECOND.value


def channel_user_map(metadata) -> Dict[str, str]:
    nums = {"L": 0, "R": 1}
    return {
        f"ch_{nums[speaker['channel']]}": speaker["user_id"]
        for speaker in metadata["speakers"]
    }


def load_conversation_tokens(convo_path: Path):
    """
    Iterate over conversations in the data download, load the token-by-token JSON data, and
    create a long-form dataframe of tokens with speaker IDs
    """
    with open(convo_path / "metadata.json", "r") as md:
        metadata = json.load(md)

    with open(convo_path / "transcription" / "transcribe_output.json", "r") as tr:
        channels = json.load(tr)

    speaker_map = channel_user_map(metadata)

    speaker_token_dfs = []

    for channel in channels["results"]["channel_labels"]["channels"]:
        tokens = pd.json_normalize(channel["items"])
        tokens["speaker"] = speaker_map[channel["channel_label"]]
        speaker_token_dfs.append(tokens)

    dfs = []
    for speaker_df in speaker_token_dfs:
        speaker_df["confidence"] = speaker_df.alternatives.map(
            lambda x: x[0]["confidence"]
        ).astype(float)
        speaker_df["utterance"] = speaker_df.alternatives.map(lambda x: x[0]["content"])
        # forward fill start/end time so that punctuation also have timestamps
        speaker_df["start"] = speaker_df.start_time.ffill().astype("float")
        speaker_df["stop"] = speaker_df.end_time.ffill().astype("float")
        # add an arbitrary small value to start_time/stop_time so punctuation follows words during sorting
        speaker_df["start_time"] = np.where(
            speaker_df.type == "punctuation",
            speaker_df.start + 0.0001,
            speaker_df.start,
        )
        speaker_df["stop_time"] = np.where(
            speaker_df.type == "punctuation", speaker_df.stop + 0.0001, speaker_df.stop
        )
        dfs.append(speaker_df)

    speakers_together = pd.concat(dfs).sort_values("start_time").reset_index(drop=True)
    speakers_together.index.name = "turn_id"

    speakers_together["conversation_id"] = convo_path.parts[-1]
    speakers_together.set_index("conversation_id", append=True, inplace=True)

    return speakers_together[
        ["speaker", "start", "stop", "utterance", "confidence", "type"]
    ].reorder_levels(["conversation_id", "turn_id"])


def heldneredlund(speaker_df, second_gap: float = None):
    """
    The crux of the Heldner and Edlund model is the joining of same-speaker utterances
    if a threshold of silence is met. Specifically, they propose that two within-speaker
    utterances should be joined if the pause between is less than 20ms.

    Parameters
    ----------
    speaker_df : pd.DataFrame
    second_gap : float
        The minimum within-speaker pause before joining utterances

    Returns
    -------
    turn_id : pd.Series
    """
    if not second_gap:
        raise Exception(
            "A second gap must be defined in the Heldner & Edlund model. Recommended: 20ms"
        )

    within_speaker_pauses = speaker_df.start - speaker_df.shift(1).stop

    # cumsum over a boolean Series results in turns being collapsed
    # with their prior if they were separated by less than second_gap
    return (within_speaker_pauses > second_gap).cumsum().astype(int)


def aggregate_tokens_into_turns(token_rows):
    """
    Transform a DataFrame of tokens into a single turn row
    """
    return pd.Series(
        {
            "start": token_rows["start"].min(),
            "stop": token_rows["stop"].max(),
            "utterance": " ".join(token_rows["utterance"]),
        }
    )


def token_to_turns(speaker_turns):
    """
    Join tokens that are separated by 20ms or less of silence
    """
    turn_ixs = heldneredlund(speaker_turns, MAXIMUM_PAUSE)
    return speaker_turns.groupby(turn_ixs).apply(aggregate_tokens_into_turns)


def load_heldner_edlund_transcript(convo_path: Path):

    convo_id = convo_path.parts[-1]

    # load the tokens for this convo and subset the columns
    tokens = load_conversation_tokens(convo_path)
    tokens = tokens[["speaker", "start", "stop", "utterance", "confidence"]]

    # collapse tokens in turns
    turns = (
        tokens.groupby("speaker")
        .apply(token_to_turns)
        .sort_values("start")
        .reset_index()
    )

    del turns["level_1"]

    # reformat the index of this transcript
    turns.index.name = "turn_id"
    turns["conversation_id"] = convo_id
    turns.set_index("conversation_id", append=True, inplace=True)
    turns = turns.reorder_levels(["conversation_id", "turn_id"])

    # add a column to indicate the duration of the turn
    turns["delta"] = (turns["stop"] - turns["start"]).round(4)

    return turns


def load_other_transcript(convo_path, model):
    transcript_csv_fn = convo_path / "transcription" / f"transcript_{model.value}.csv"
    transcript_df = pd.read_csv(transcript_csv_fn)
    transcript_df["conversation_id"] = convo_path.parts[-1]
    return transcript_df


def collate_transcripts(model):
    """
    iterate over each conversation, load one of its transcripts, and join all
    transcripts into a single data frame
    """

    data_download_glob = [f for f in Path(DATA_DOWNLOAD_PATH).glob("*") if f.is_dir()]
    model_str = model.value
    output_fn = f"{TRANSCRIPT_OUTPUT_PREFIX}{model_str}.csv"

    if model == Model.HELDNEREDLUND:
        # Heldner-Edlund transcripts are created from the token-level data
        transcripts = pqdm(
            data_download_glob, load_heldner_edlund_transcript, n_jobs=N_JOBS
        )
        transcripts = pd.concat(transcripts, axis="index")

    elif model in (Model.AUDIOPHILE, Model.BACKBITER, Model.CLIFFHANGER):
        # Other transcripts are precomputed and can be loaded
        data_download_glob = [(path, model) for path in data_download_glob]
        transcripts = pqdm(
            data_download_glob,
            load_other_transcript,
            argument_type="args",
            n_jobs=N_JOBS,
        )
        transcripts = pd.concat(transcripts, axis="index")
        # move the convo id column to the far left
        transcripts.insert(0, "conversation_id", transcripts.pop("conversation_id"))

    transcripts.to_csv(output_fn)


def set_turn_relativity(conversation):
    """
    implementation of the Heldner & Edlund (2010) state classification algorithm
    """

    # first step is to create a new matrix with every column spanning an interval of time (e.g. 10ms)
    last_utterance_time = conversation["stop_granular"].max()
    speaking = np.zeros((2, last_utterance_time), dtype=np.uint8)

    # "shade" in the columns of where speech is happening

    speaker_index = 0
    speaker_order = []
    for g, speaker in conversation.groupby("speaker"):
        speaker_order.append(g)
        for row in speaker.itertuples():
            speaking[speaker_index, row.start_granular : row.stop_granular] = 1

        speaker_index += 1

    # do some boolean logic to define the states of each time frame

    speech_state_classifications = np.zeros((4, last_utterance_time), dtype=np.uint8)
    speech_state_classifications[0, :] = (speaking[0, :] ^ speaking[1, :]) & speaking[
        0, :
    ]  # is only S1 speaking?
    speech_state_classifications[1, :] = (speaking[0, :] ^ speaking[1, :]) & speaking[
        1, :
    ]  # is only S2 speaking?
    speech_state_classifications[2, :] = (
        speaking[0, :] & speaking[1, :]
    )  # are both speakers speaking?
    speech_state_classifications[3, :] = (speaking[0, :] == 0) & (
        speaking[1, :] == 0
    )  # is there silence?

    # get an integer (1, 2, 3, or 4) of the state of each frame
    states = speech_state_classifications.argmax(axis=0)

    transition_history = []

    # iterate over each state frame
    for frame, state in enumerate(states):

        current_transition = {
            "speaker": None,
            "state": state,
            "frame": frame,
            "classification": None,
        }

        # if the transition history is empty, start it off with the first frame's state
        if len(transition_history) == 0:
            transition_history.append(current_transition)
            continue

        # otherwise, keep track of what the most recent state was
        ultimate_state = transition_history[-1]["state"]

        # skip this frame if there was no state transition
        if state == ultimate_state:
            continue

        # if we can look back in time 2 states and someone is now speaking
        if len(transition_history) > 2 and state in [0, 1]:

            penultimate_state = transition_history[-2]["state"]

            # if the 2nd most recent state is the same as the current state,
            # then the current transition can only either be a WSO or a pause
            if penultimate_state == state:
                # no speaker change

                if ultimate_state == 2:
                    # within-speaker overlap (interrupting)
                    current_transition["classification"] = "wso"
                elif ultimate_state == 3:
                    # within-speaker silence, pause
                    current_transition["classification"] = "pause"
            elif penultimate_state == 1 - state:
                # if the 2nd most recent state is the opposite of the current
                # state, then the current transition can only either be an
                # overlap or a gap

                if ultimate_state == 2:
                    # between-speaker overlap
                    current_transition["classification"] = "overlap"
                elif ultimate_state == 3:
                    # between-speaker silence, gap
                    current_transition["classification"] = "gap"

            ultimate_frame = transition_history[-1]["frame"]
            current_transition["prev_frame"] = ultimate_frame
            current_transition["duration"] = (frame - ultimate_frame) / GRANULARITY
            current_transition["speaker"] = speaker_order[state]

        transition_history.append(current_transition)

    df = pd.DataFrame(transition_history)
    # the df will contain many non-transition events as null, so drop those
    return df.loc[~df["classification"].isnull()].reset_index(drop=True)


def create_turn_exchanges(model):
    """
    with a transcript, apply the state classification algorithm and clean up
    the output
    """

    model_str = model.value
    input_fn = f"{TRANSCRIPT_OUTPUT_PREFIX}{model_str}.csv"
    output_fn = f"{TURN_EXCHANGE_OUTPUT_PREFIX}{model_str}.csv"

    transcripts = pd.read_csv(input_fn)

    # transform the start and stop to integer valued columns
    transcripts["start_granular"] = (
        (transcripts.loc[:, "start"] * GRANULARITY).round().astype(int)
    )
    transcripts["stop_granular"] = (
        (transcripts.loc[:, "stop"] * GRANULARITY).round().astype(int)
    )

    # for each conversation, apply the Heldner and Edlund algorithm
    tqdm.pandas(desc="CONVERSATIONS")
    raw_intervals = transcripts.groupby("conversation_id").progress_apply(
        set_turn_relativity
    )

    # add the speaker ID as a new index, to create a MultiIndex
    raw_intervals.set_index("speaker", append=True, inplace=True)

    # Overlaps are inherently negative values
    raw_intervals.loc[lambda x: x.classification == "overlap", "duration"] *= -1
    raw_intervals["prev_frame"] = raw_intervals["prev_frame"].round().astype(int)
    raw_intervals["frame"] = raw_intervals["frame"].round().astype(int)

    # reformat the index names
    raw_intervals.index.names = ["conversation_id", "turn_exchange_id", "speaker"]

    raw_intervals.to_csv(output_fn)


def load_audio_video_features(convo_path):
    audio_video_df = pd.read_csv(convo_path / "audio_video_features.csv")
    audio_video_df["conversation_id"] = convo_path.parts[-1]
    return audio_video_df


def collate_audio_video_features():
    """
    iterate over each conversation, load its survey data, and join all survey
    responses into a single data frame
    """

    data_download_glob = [f for f in Path(DATA_DOWNLOAD_PATH).glob("*") if f.is_dir()]
    output_fn = DATA_PATH / "audio_video_features.csv"

    audio_video_features = pqdm(data_download_glob, load_audio_video_features, n_jobs=N_JOBS)
    audio_video_features = pd.concat(audio_video_features, axis="index")

    audio_video_features.to_csv(output_fn)


def load_survey(convo_path):
    return pd.read_csv(convo_path / "survey.csv")


def collate_surveys():
    """
    iterate over each conversation, load its survey data, and join all survey
    responses into a single data frame
    """

    data_download_glob = [f for f in Path(DATA_DOWNLOAD_PATH).glob("*") if f.is_dir()]
    output_fn = DATA_PATH / "surveys.csv"

    surveys = pqdm(data_download_glob, load_survey, n_jobs=N_JOBS)
    surveys = pd.concat(surveys, axis="index")

    surveys.rename({'convo_id': 'conversation_id'}, axis='columns', inplace=True)

    surveys.to_csv(output_fn)


def create_embeddings(sbert_model, model_prefix):
    embedder = SentenceTransformer(sbert_model)
    D = embedder.get_sentence_embedding_dimension()

    transcripts = pd.read_csv(f"{TRANSCRIPT_OUTPUT_PREFIX}{Model.BACKBITER.value}.csv")
    N = len(transcripts)

    embeddings = np.empty((N, D))
    for i in tqdm(range(N)):
        embeddings[i] = embedder.encode(transcripts[['utterance']].iat[i, 0], convert_to_numpy=True)

    embeddings_df = pd.DataFrame(
        embeddings,
        columns = [model_prefix + '_' + str(i).zfill(4) for i in range(D)]
    )

    merged = pd.concat([transcripts, embeddings_df], axis = 1)
    merged.to_csv(DATA_PATH / f"transcripts-{model_prefix}-embeddings--backbiter.csv")


if __name__ == "__main__":

    print("Collate survey responses")
    collate_surveys()

    print("Collate audio/video features")
    collate_audio_video_features()

    # for each of the four models, save a trancript CSV and a turn exchange CSV
    for model in [
        Model.HELDNEREDLUND,
        Model.AUDIOPHILE,
        Model.BACKBITER,
        Model.CLIFFHANGER,
    ]:
        print(f"Create {model} transcripts")
        collate_transcripts(model)

        print(f"Create {model} turn exchanges")
        create_turn_exchanges(model)
    
    print("Create MPNet embeddings")
    create_embeddings('all-mpnet-base-v2', 'mpnet')

    print("Create RoBERTa embeddings")
    create_embeddings('roberta-large-nli-stsb-mean-tokens', 'roberta')
