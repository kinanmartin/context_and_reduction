from tqdm import tqdm
# from tqdm.notebook import tqdm
from datasets import load_dataset, DatasetDict, load_from_disk
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel, GPT2LMHeadModel
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import random


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("input_data_path")
    parser.add_argument("tokenized_data_path")
    parser.add_argument("need_to_tokenize")
    parser.add_argument("model_output_path")

    parser.set_defaults(

    )

    args = parser.parse_args()

    input_data_path = args.input_data_path
    tokenized_data_path = args.tokenized_data_path
    need_to_tokenize = args.need_to_tokenize

    # Initialize tokenizer
    context_length = 1024
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Load (pre)tokenized data
    TOKENIZE = sys.argv[0]
    if TOKENIZE: 
        pass
