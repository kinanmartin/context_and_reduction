import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"

def tokenize_cliffhanger_turn(text, tokenizer):
    """
    Tokenizes a CANDOR cliffhanger turn. Text will be split
    on whitespace if it is not already.

    Check: is tokenization with is_split_into_words the same
    as tokenization split on whitespace? If not, I should join
    "text" by whitespace back into a single string
    """
    if isinstance(text, str):
        text = text.split(' ')
    inputs = tokenizer(
        text, 
        return_tensors='pt',
        return_offsets_mapping=True, 
        add_special_tokens=True,
        is_split_into_words=True, 
    )
    # print(inputs)
    # for i, word in enumerate(text):
    #     print(f'{word=}, {inputs.word_to_tokens(i)}') 
    return inputs

# def cliffhanger_df_to_tokens(cliffhanger_df, tokenizer):
#     cliffhanger_df['tokens'] = cliffhanger_df['utterance'].apply(
#         lambda text: tokenize_cliffhanger_turn(text, tokenizer)
#     )
#     return cliffhanger_df


def calculate_surprisal(inputs, model):
    """
    Given a tokenized encoding `inputs` of type 
        transformers.tokenization_utils_base.BatchEncoding

    Return a tensor of surprisals for each token.

    Note that len(surprisals) is one less than len(inputs)
    because of the bos_token: the surprisal value of at
    index 0 will be hardcoded to float('inf') to match lengths

    TODO: implement has_bos_token arg
    """
    # print(inputs)
    input_ids = inputs['input_ids']

    with torch.no_grad():
        input_ids = input_ids.to(device)
        outputs = model(input_ids, labels=input_ids)
        logit_predictions = outputs.logits
        
        shift_logits = logit_predictions[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        surprisals = loss.view(shift_labels.size()).cpu().numpy().tolist()
        
    surprisals = surprisals[0]
    surprisals.insert(0, float('inf'))
    return surprisals

def aggregate_surprisal_by_word(inputs, surprisals):
    """
    Aggregate (add) surprisals of tokens which come
    from the same word
    """
    inputs_words = inputs.word_ids()
    n_words = (inputs_words[-1] + 1) # better way?
    # print(n_words, len(inputs_words), len(surprisals))
    surprisals_by_word = [0] * n_words
    for token_idx, word_idx in enumerate(inputs_words):
        surprisals_by_word[word_idx] += surprisals[token_idx]
    return surprisals_by_word
    # for i, word in enumerate(text):
    #     print(f'{word=}, {inputs.word_to_tokens(i)}') 


if __name__ == '__main__':
    model = load_pretrained_model('gpt2')
    tokenizer = load_pretrained_tokenizer('gpt2', add_prefix_space=True)
    model.resize_token_embeddings(len(tokenizer))

    
    # text = "That's awesome. When I, like the last time I was in Wisconsin was like several years ago and one of the best meals that I had, there was actually at a fish fry, like a friday night fish fry. This place called the serb hall in Milwaukee and it was like the strangest environment to eaten ever. It was like very large event hall and all of the wait staff were like wearing very formal suits and tuxedos, But like the dress code of the actual people eating, there was very relaxed and when I say 40% of the people were wearing like exactly what you're wearing. I am not exaggerating. And so you were surrounded by like wait staff wearing like either tuxedos are very formal suits and then like green bay packers gear as far as the eye can see and the Munich selection ranged from like Sarah McLachlan to traditional Serbian folk music and like it was, it was very strange but the food was incredible and they also had like spotted cow as well too. So I had as much spotted cow because we don't have that in Iowa. So I was drinking as much spotted how could possibly consume and as much fried fish and I'm still chasing that high because that was so are the best food I have ever had. And so I don't know, I just want to say Wisconsin, I about to your prowess of making incredible fried fish. Um So yeah, I don't know. I don't know why. I just looked like, like I said, really, I cannot stop thinking about that meal. It's like every couple of months man, like, you know, it was really good."
    text = "I don't like it. I do n't like it ."
    inputs = tokenize_cliffhanger_turn(text, tokenizer)
    print(inputs['input_ids'])
    surprisals = calculate_surprisal(inputs, model)
    surprisals_by_word = aggregate_surprisal_by_word(inputs, surprisals)

    text_split = text.split(' ')
    assert len(text_split) == len(surprisals_by_word), "lengths don't match!"
    for word, surprisal in zip(text_split, surprisals_by_word):
        print(word, surprisal)

    
