# from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_from_disk
import torch
from tqdm import tqdm
from utils import *

from torch.nn import CrossEntropyLoss
import torch

# from torch.utils.data import DataLoader

# from evaluate import load

from torch.utils.data import DataLoader


def calculate_perplexity(model, tokenizer, data_collator, test_dataset, batch_size=64):
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=data_collator)
    
    total_loss = 0
    total_examples = 0

    # Disable gradient calculation for efficiency and to prevent memory issues
    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs = {key: val.to(model.device) for key, val in batch.items() if key in ['input_ids', 'attention_mask', 'labels']}
            outputs = model(**inputs)
            loss = outputs.loss
            batch_loss = loss.item() * inputs["input_ids"].size(0)
            batch_size = inputs["input_ids"].size(0)
            print(batch_loss)
            print(batch_size)
            total_loss += batch_loss
            total_examples += batch_size
            print(total_loss)
            print()

    # Calculate the mean loss over all batches
    mean_loss = total_loss / total_examples
    
    # Perplexity is the exponential of the cross-entropy (mean loss)
    perplexity = torch.exp(torch.tensor(mean_loss))

    return perplexity.item()

# def calculate_perplexity(model, tokenizer, dataset, device, batch_size=32):
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         model = model.to(device)

#         # if batch_size > 1 (which generally leads to padding being required), and
#         # if there is not an already assigned pad_token, assign an existing
#         # special token to also be the padding token
#         # if tokenizer.pad_token is None and batch_size > 1:
#         #     existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
#         #     # check that the model already has at least one special token defined
#         #     assert (
#         #         len(existing_special_tokens) > 0
#         #     ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
#         #     # assign one of the special tokens to also be the pad token
#         #     tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

#         # if add_start_token and max_length:
#         #     # leave room for <BOS> token to be added:
#         #     assert (
#         #         tokenizer.bos_token is not None
#         #     ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
#         #     max_tokenized_len = max_length - 1
#         # else:
#         #     max_tokenized_len = max_length

#         predictions = [x['text'] for x in dataset]

#         encodings = tokenizer(
#             predictions,
#             add_special_tokens=False,
#             padding=True,
#             # truncation=True if max_tokenized_len else False,
#             # max_length=max_tokenized_len,
#             return_tensors="pt",
#             return_attention_mask=True,
#         ).to(device)

#         encoded_texts = encodings["input_ids"]
#         attn_masks = encodings["attention_mask"]

#         # check that each input is long enough:
#         # if add_start_token:
#         #     assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
#         # else:
#         #     assert torch.all(
#         #         torch.ge(attn_masks.sum(1), 2)
#         #     ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

#         ppls = []
#         loss_fct = CrossEntropyLoss(reduction="none")

#         for start_index in tqdm(range(0, len(encoded_texts), batch_size)):
#             end_index = min(start_index + batch_size, len(encoded_texts))
#             encoded_batch = encoded_texts[start_index:end_index]
#             attn_mask = attn_masks[start_index:end_index]

#             # if add_start_token:
#             #     bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(device)
#             #     encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
#             #     attn_mask = torch.cat(
#             #         [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device), attn_mask], dim=1
#             #     )

#             labels = encoded_batch

#             with torch.no_grad():
#                 out_logits = model(encoded_batch, attention_mask=attn_mask).logits

#             shift_logits = out_logits[..., :-1, :].contiguous()
#             shift_labels = labels[..., 1:].contiguous()
#             shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

#             perplexity_batch = torch.exp(
#                 (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
#                 / shift_attention_mask_batch.sum(1)
#             )

#             print(perplexity_batch)
#             print(np.mean(perplexity_batch))
#             ppls += perplexity_batch.tolist()

#         return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}

# def calculate_perplexity(model, predictions):
#     perplexity = load("perplexity", module_type="metric")
#     results = perplexity.compute(predictions, model_id)


# def calculate_perplexity(model, tokenizer, dataset):
#     model.eval()
#     model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
#     loss_fct = CrossEntropyLoss(reduction="none")

#     with torch.no_grad():
#         for example in tqdm(dataset):
#             input_ids = torch.tensor(example['input_ids']).unsqueeze(0).to(model.device)
#             attention_mask = torch.tensor(example['attention_mask']).unsqueeze(0).to(model.device)
            
#             outputs = model(input_ids, labels=input_ids, attention_mask=attention_mask)

#             logits = outputs.logits[:, :-1, :].contiguous()
#             labels = input_ids[:, 1:].contiguous()


#             loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
#             # print(loss.item())
#             print(loss.item() * labels.numel())
#             # print(labels.numel())
            
#             total_loss += loss.item() * labels.numel()
#             total_length += labels.numel()

#             i += 1
#             if i == 200:
#                 break



# def calculate_perplexity(model, dataset):
#     model.eval()
#     model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
#     total_loss = 0
#     total_length = 0
    
#     i = 0
#     with torch.no_grad():
#         for example in tqdm(dataset):
#             input_ids = torch.tensor(example['input_ids']).unsqueeze(0).to(model.device)
#             attention_mask = torch.tensor(example['attention_mask']).unsqueeze(0).to(model.device)
#             if input_ids.size() == 0:
#                 continue
            
#             try:
#                 outputs = model(input_ids, labels=input_ids, attention_mask=attention_mask)
#             except:
#                 continue
#             logits = outputs.logits[:, :-1, :].contiguous()
#             labels = input_ids[:, 1:].contiguous()

#             loss_fct = CrossEntropyLoss()
#             loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
#             # print(loss.item())
#             print(loss.item() * labels.numel())
#             # print(labels.numel())
            
#             total_loss += loss.item() * labels.numel()
#             total_length += labels.numel()

#             i += 1
#             if i == 200:
#                 break

#     average_loss = total_loss / total_length
#     perplexity = torch.exp(torch.tensor(average_loss))
#     return perplexity.item()




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
    test_set = test_set.remove_columns('text')
    print('...done')

    perplexity = calculate_perplexity(model, tokenizer, data_collator, test_set)
    print(f"Perplexity: {perplexity}")

    # from coca_tokenize import load_data_in_splits
    # coca_dir = "data/coca_spoken/text_sentence_cleaned/"

    # coca_dsdict = load_data_in_splits(coca_dir, .8, .1, .1)
    # test_set = coca_dsdict['test']

    # testset_dir = "data/coca_spoken/text_sentence_cleaned"
    # print(f'Loading {testset_dir=}...')
    # test_set = load_dataset(testset_dir)
    # print('...done')

    # example_batch = test_set[0:5]
    # print(example_batch['input_ids'])  # Check the first few batches for structure
    # print(example_batch['attention_mask'])

    # perplexity = calculate_perplexity(model, tokenizer, test_set, 'cpu')
    # print(f"Perplexity: {perplexity}")

