# from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_from_disk
import torch
from tqdm import tqdm
from utils import *

from torch.nn import CrossEntropyLoss
import torch

from torch.utils.data import DataLoader

def calculate_perplexity(model, dataset):
    model.eval()
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    total_loss = 0
    total_length = 0
    
    with torch.no_grad():
        for example in tqdm(dataset):
            input_ids = torch.tensor(example['input_ids']).unsqueeze(0).to(model.device)
            attention_mask = torch.tensor(example['attention_mask']).unsqueeze(0).to(model.device)
            
            outputs = model(input_ids, labels=input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, :-1, :].contiguous()
            labels = input_ids[:, 1:].contiguous()

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            total_loss += loss.item() * labels.numel()
            total_length += labels.numel()

    average_loss = total_loss / total_length
    perplexity = torch.exp(torch.tensor(average_loss))
    return perplexity.item()




# def calculate_perplexity(model, dataset, collator):
#     model.eval()  # Set model to evaluation mode
#     total_loss = 0
#     total_length = 0
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
    
#     dataloader = DataLoader(dataset, batch_size=8, collate_fn=collator, shuffle=False)

#     with torch.no_grad():
#         for batch in dataloader:
#             inputs = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels = inputs.clone()

#             outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
#             shift_logits = outputs.logits[:, :-1, :].contiguous()
#             shift_labels = labels[:, 1:].contiguous()

#             loss_fct = CrossEntropyLoss()
#             loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
#             total_loss += loss.item() * shift_labels.numel()
#             total_length += shift_labels.numel()

#     average_loss = total_loss / total_length
#     perplexity = torch.exp(torch.tensor(average_loss))
#     return perplexity.item()



# def calculate_perplexity(model, tokenizer, encodings, stride=512):
#     """
#     From https://huggingface.co/docs/transformers/perplexity
#     """

#     # encodings = tokenizer("\n\n".join(test_set["text"]), return_tensors="pt")

#     max_length = model.config.n_positions
#     stride = 512
#     seq_len = encodings.input_ids.size(1)

#     nlls = []
#     prev_end_loc = 0
#     for begin_loc in tqdm(range(0, seq_len, stride)):
#         end_loc = min(begin_loc + max_length, seq_len)
#         trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
#         input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
#         target_ids = input_ids.clone()
#         target_ids[:, :-trg_len] = -100

#         with torch.no_grad():
#             outputs = model(input_ids, labels=target_ids)

#             # loss is calculated using CrossEntropyLoss which averages over valid labels
#             # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
#             # to the left by 1.
#             neg_log_likelihood = outputs.loss

#         nlls.append(neg_log_likelihood)

#         prev_end_loc = end_loc
#         if end_loc == seq_len:
#             break

#     ppl = torch.exp(torch.stack(nlls).mean())

#     return ppl


if __name__ == "__main__":
    from datasets import disable_caching
    disable_caching()

    # model_dir = "models/script1/left_sentence/"
    model_dir = "gpt2"
    tokenizer_name = "gpt2"

    model = load_pretrained_model(model_dir)
    tokenizer = load_pretrained_tokenizer(tokenizer_name)
    data_collator = init_data_collator(tokenizer, 'left')

    tokenized_testset_dir = "data/coca_spoken/tokens_sentence/test"

    print(f'Loading {tokenized_testset_dir=}...')
    test_set = load_from_disk(tokenized_testset_dir)
    print('...done')

    # example_batch = test_set[0:5]
    # print(example_batch['input_ids'])  # Check the first few batches for structure
    # print(example_batch['attention_mask'])

    perplexity = calculate_perplexity(model, test_set)
    print(f"Perplexity: {perplexity}")

