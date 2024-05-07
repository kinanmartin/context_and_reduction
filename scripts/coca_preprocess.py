import random
import re
from typing import List
from pathlib import Path
from tqdm import tqdm

from datasets import load_dataset

from nltk.tokenize import TreebankWordDetokenizer
# from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk import sent_tokenize

def separate_chunks(text: str) -> List[str]:
    """
    COCA is composed of scrambled chunks split by "@" * 10 (possibly 
    cut off at end of file). 
    Returns a list of separated chunks.
    """
    return text.split(' @ @ @ @ @ @ @ @ @ @ ')

def remove_speaker_and_other_tags(chunk: str, remove_nonspeaker_tags=True) -> str:
    """
    DEPRECATED: it's better to split text by these tags instead of removing them
    Remove from one chunk speaker tags (ex: @!BOB:) and optionally
    other tags (ex: @(End-of-clip)).
    """
    pattern = r"\s+@\S+" if remove_nonspeaker_tags else r"\s+@!\S+"
    return re.sub(pattern, " . ", chunk)

def split_by_speaker_and_other_tags(
        chunk: str, 
        remove_nonspeaker_tags=True,
        ) -> List[str]:
    """
    Splits one chunk by speaker tags (ex: @!BOB) and optionally
        other tags (ex: @(End-of-clip)).

    remove_nonspeaker_tags: also removes things like @(End-of-clip). 
        speaker tag: @!BOB  non-speaker tag @BOB (no "!")
        Does not remove long portions inside of @(Clip-from-previous blocks
    
    Notes:
        - Pattern makes first word in turn start with a space.
        To remove it, add an \s at the end of the pattern, but be aware
        that this will break pattern matching of consecutive tags.
        - Speaker tags are inconsistently either marked as 
            "@!BOB", "@!BOB:", "@!BOB :", "@!BOB ( voiceover ) :", 
            and more. ( voiceover ) is currently not captured.

    """
    # pattern = r"\s+@\S+" if remove_nonspeaker_tags else r"\s+@!\S+"
    pattern = r"@\S+(?:\s:|)\s" if remove_nonspeaker_tags else r"@!\S+(?:\s:|)\s"
    out = re.split(pattern, chunk)
    out = [segment for segment in out if segment.strip()]
    return out

def split_turn_into_sentences(
        turn: str, 
        # exclude_sentences_with_ellipses=False
        ) -> List[str]:
    """
    DEPRECATED: use nltk sent_tokenize instead.
    Splits one tag-free turn (as separated by split_by_speaker_and_other_tags) 
        into sentences.
    Since COCA has space-separated punctuation, splits are done by:
        [' . ', ' ? ', ' ! ']
    """
    delimiters = [' . ', ' ? ', ' ! ']
    pattern = "|".join(map(re.escape, delimiters))
    pattern = '(' + pattern + ')' # retain delimiters
    splits = re.split(pattern, turn)
    if len(splits) == 1:
        return splits
    
    # For multi-sentence utterances, we must manually re-combine punctuation
    out = []
    for idx, split in enumerate(splits):
        if not split:
            continue
        if not (idx % 2): # is sentence
            out.append(split)
        else: # is delimiter
            out[-1] += split[:-1] # don't include space after punctuation
    return out

def nltk_split_turn_into_sentences(turn):
    return sent_tokenize(turn)

def split_chunk_into_sentences(
        chunk: str,
        exclude_first_and_last_sentences=True,
        remove_nonspeaker_tags=True,
        nltk_sent_tokenize=True,
        ) -> List[str]:
    """
    Combines `split_by_speaker_and_other_tags` and 
        `split_turn_into_sentences` to split a COCA chunk
        into a list of sentences.

    exclude_first_and_last_sentences: because the first and 
        last sentences are likely fragments split by the chunk border
    """
    turns = split_by_speaker_and_other_tags(chunk, 
                                            remove_nonspeaker_tags)
    sentences = []
    for turn in turns:
        if nltk_sent_tokenize:
            sentences.extend(nltk_split_turn_into_sentences(turn))
        else:
            sentences.extend(split_turn_into_sentences(turn))
    return sentences[1:-1] if exclude_first_and_last_sentences else sentences


def clean_coca_file(
        input_file_path: Path,
        output_dir_path: Path,
        split_by='chunk',
        overwrite=True,
        exclude_first_and_last_sentences=True,
        remove_nonspeaker_tags=True,
        nltk_detokenize=True,
        replace_emdash_with_comma=True,
        ) -> None:
    if nltk_detokenize:
        detokenizer = TreebankWordDetokenizer()

    context_choices = ['bigram', 'sentence', 'chunk']
    assert input_file_path.exists(), f'File "{input_file_path}" not found'
    assert split_by in context_choices, f'Invalid split method {split_by}: choose from {context_choices}'

    dataset_dict = load_dataset('text', data_files=str(input_file_path))
    dataset = dataset_dict['train']

    output_dir_path.mkdir(parents=True, exist_ok=overwrite)
    output_file_path = output_dir_path / (input_file_path.stem + '_cleaned.txt')

    f = open(output_file_path, 'w')
    for line in tqdm(dataset):
        text = line['text']
        chunks = separate_chunks(text)
        if split_by == 'chunk':
            f.write('\n'.join(chunks) + '\n')
        else:
            for chunk in chunks:
                sentences = split_chunk_into_sentences(chunk,
                                                    exclude_first_and_last_sentences,
                                                    remove_nonspeaker_tags)
                if nltk_detokenize:
                    sentences = [detokenizer.detokenize(sentence.split(' ')) for sentence in sentences]
                    if replace_emdash_with_comma:
                        sentences = [sentence.replace('--', ', ') for sentence in sentences]

                if split_by == 'sentence':
                    f.write('\n'.join(sentences) + '\n')
                elif split_by == 'bigram':
                    for sentence in sentences:
                        words = sentence.split(' ')
                        f.write('<s>' + ' ' + words[0] + '\n')
                        for i in range(1, len(words)-1):
                            f.write(words[i] + ' ' + words[i+1] + '\n')
                        f.write(words[-1] + ' ' + '</s>' '\n')
        

    f.close()
    return None


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()
    # parser.add_argument("--input_data_dir")
    # parser.add_argument("--need_to_tokenize")
    parser.add_argument("--coca_dir", type=Path)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--context_size", default='sentence')

    args = parser.parse_args()


    # coca_dir = "data/coca/text/text_spoken_kde/"
    coca_dir = args.coca_dir
    # output_dir = Path("data/coca_spoken/text_bigram_cleaned/")
    output_dir = args.output_dir
    context_size = args.context_size

    # clean_coca_file(
    #     input_file_path=Path("../data/coca/text/text_spoken_kde/w_spok_2000.txt"),
    #     output_dir_path=Path("../data/coca_spoken/text_chunk_cleaned/"),
    #     split_by='chunk'
    # )

    for file in coca_dir.iterdir():
        clean_coca_file(
            input_file_path=file, 
            output_dir_path=output_dir,
            split_by=context_size
        )

    # # dataset = load_dataset('text', data_dir=coca_dir)
    # dataset = load_dataset('text', data_files=coca_dir+'w_spok_201*.txt')
    # train_dataset = dataset['train']

    # example_line = random.choice(train_dataset)
    # print(example_line['text'][:100])

    # example_string_id = random.randint(0, len(train_dataset) - 1)
    # example_string = train_dataset[example_string_id]['text']
    # print(f'{example_string_id=}')
    # print(f'{len(example_string)=}')
    # print(example_string[:100])


    # example_chunks = separate_chunks(example_string)
    # print(len(example_chunks), [len(chunk) for chunk in example_chunks])
    # for chunk in example_chunks:
    #     print(chunk[:50])

    # example_chunk_id = random.randint(0, len(example_chunks)-1)
    # print(f'{example_chunk_id=}')
    # example_chunk = example_chunks[example_chunk_id]
    # print(example_chunk)
    # print(remove_speaker_and_other_tags(example_chunk))
    # print('----')
    # example_turns = split_by_speaker_and_other_tags(example_chunk)
    # for turn_number, turn in enumerate(example_turns):
    #     print(turn_number, turn)

    # turn = example_turns[random.randint(0, len(example_turns)-1)]
    # split_turn_into_sentences(turn)

    # chunk_number = random.randint(0, len(example_chunks)-1)
    # example_chunk = example_chunks[chunk_number]
    # example_sentences = split_chunk_into_sentences(example_chunk, 
    #                                             exclude_first_and_last_sentences=True)
    # print(f'{chunk_number=}')
    # example_sentences
    # print(example_chunks[chunk_number])