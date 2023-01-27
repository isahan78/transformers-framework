
# This code defines a function predict(input_text, k=5, ngram_model=ngram_model) that takes in an input text, 
# encodes it into token indices, generates the input mask, passes the input through the model, converts the logits to 
# probabilities, performs beam search with a beam size of k and uses n-gram model to estimate the probability of a sequence.

import torch
from torch.nn.functional import softmax
from model import Transformer
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE
from collections import Counter

# Instantiate the model
model = Transformer(d_model=512, nhead=8, num_layers=6, dim_feedforward=2048)

# Load the trained model parameters
model.load_state_dict(torch.load("trained_model.pt"))

# Move the model to the evaluation mode
model.eval()

# Generate the n-grams
n = 3
train_data, padded_sents = padded_everygram_pipeline(n, train_text)
ngram_model = MLE(n)
ngram_model.fit(train_data, padded_sents)

# define a function to predict
def predict(input_text, k=5, ngram_model=ngram_model):
    # Encode the input text
    input_text = torch.tensor(tokenizer.encode(input_text)).unsqueeze(0)
    input_text = input_text.to(device)

    # Generate the input mask
    input_mask = ...

    # Forward pass
    with torch.no_grad():
        output = model(input_text, input_mask)

    # Convert the logits to probabilities
    prob = softmax(output, dim=-1)

    # Beam search
    beam = [(input_text, 0)]
    for _ in range(k):
        candidates = []
        for seq, score in beam:
            next_token_logprob = prob[:, seq[-1]]
            ngram_logprob = ngram_model.score(seq)
            total_logprob = next_token_logprob + ngram_logprob
            best_candidates = total_logprob.topk(k)
            for i, candidate_logprob in enumerate(best_candidates.values):
                candidate = torch.cat([seq, best_candidates.indices[i].unsqueeze(0)])
                candidates.append((candidate, score + candidate_logprob))
        beam = sorted(candidates, key=lambda x: x[1], reverse=True)[:k]

    # Decode the token indices to text
    next_token = tokenizer.decode([int(i) for i in beam[0][0]])
    return next_token

# Test the function
input_text = "Hello, how are you today?"
print(predict(input_text))
