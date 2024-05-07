# from tqdm import tqdm
# from tqdm.notebook import tqdm
from datasets import load_dataset, DatasetDict, load_from_disk
# from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel, GPT2LMHeadModel
# from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import accelerate
import transformers
# transformers.__version__, accelerate.__version__
import torch
torch.cuda.is_available()
from utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"
print(torch.cuda.get_device_name(0))

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
    parser.add_argument("--eval", default=False)
    parser.add_argument("--from_pretrained", default=False)
    parser.add_argument("--context_size", default='sentence')
    parser.add_argument("--context_direction", default='left')

    args = parser.parse_args()

    # Init/load model
    if args.from_pretrained:
        model = load_pretrained_model(args.model_dir)
    else:
        model = init_model(args.context_size)

    model.to(device)
    if args.eval:
        model.eval()

    # Load tokenizer
    tokenizer = load_pretrained_tokenizer(
        'gpt2', 
        context_size=args.context_size, 
        context_direction=args.context_direction,
        padding=True) # shouldn't change anything now
    model.resize_token_embeddings(len(tokenizer))

    # Create data collator
    data_collator = init_data_collator(tokenizer, args.context_direction)

    train_args = TrainingArguments(
        args.model_dir,
        per_device_train_batch_size=args.per_device_batch_size, # change to fit GPU specs
        per_device_eval_batch_size=args.per_device_batch_size,
        # auto_find_batch_size=True,
        evaluation_strategy='epoch',
        eval_steps=1,
        logging_steps=0.01,
        save_strategy='epoch',
        save_steps=0.25,
        # group_by_length=True if args.context_direction != 'bidi' else False,
        group_by_length=True,
        # load_best_model_at_end=True,
        # metric_for_best_model='loss',
        # greater_is_better=False,
        save_total_limit=5,
        num_train_epochs=1,
    )
    print(train_args.device)

    tokenized_dataset_dict = load_datasetdict(
        args.tokenized_data_dir,
        disable_cache=True
    )

    if not args.eval:
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=train_args,
            data_collator=data_collator,
            train_dataset=tokenized_dataset_dict['train'],
            eval_dataset=tokenized_dataset_dict['val'],
        )

        if not args.from_pretrained:
            trainer.train()
        else:
            trainer.train(
                resume_from_checkpoint=args.model_dir
            )


    


