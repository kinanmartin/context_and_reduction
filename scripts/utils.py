import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_pretrained_model(pretrained_model_name_or_path):
    model = GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path)
    model.to(device)
    model.eval()
    return model

def load_pretrained_tokenizer(pretrained_model_name_or_path, context=None):
    tokenizer = GPT2TokenizerFast.from_pretrained(
        pretrained_model_name_or_path, 
        add_prefix_space=True, # ?
    )

    if context == 'bigram':
        tokenizer.bos_token = '<s>'
        tokenizer.eos_token = '</s>'

    tokenizer.pad_token = tokenizer.eos_token # ?
    return tokenizer

if __name__ == "__main__":
    model = load_pretrained_model('gpt2')
    tokenizer = load_pretrained_tokenizer('gpt2')

