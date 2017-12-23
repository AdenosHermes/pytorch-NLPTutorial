import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models import *

torch.manual_seed(1)
use_cuda = torch.cuda.is_available()
print("with GPU:", use_cuda)
ftype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
ltype = torch.cuda.LongTensor if use_cuda else torch.LongTensor


######################################################################
# Prepare data:

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)


def prepare_char(seq, to_ix):
    idxs = [to_ix[w.lower()] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

# wording embedding
training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]
word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print(word_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 6
HIDDEN_DIM = 6

#character embedding
char_to_ix = {}
for sent, _ in training_data:
    for word in sent:
        for char in word:
            if char not in char_to_ix:
                char_to_ix[char.lower()] = len(char_to_ix)
print(char_to_ix)

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
char_EMBEDDING_DIM = 6
char_HIDDEN_DIM = 6



######################################################################
# Train the model:
model_char = charTagger(char_EMBEDDING_DIM, char_HIDDEN_DIM,
                        len(char_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer_char = optim.SGD(model_char.parameters(), lr=0.1, momentum=0.5)




model = LSTMTagger(EMBEDDING_DIM, char_HIDDEN_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.5)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
inputs = prepare_sequence(training_data[0][0], word_to_ix)
#tag_scores = model(inputs, )
#print(tag_scores)

for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        model.zero_grad()
        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        model_char.hidden = model_char.init_hidden()
        model.hidden = model.init_hidden()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Variables of word indices.

            
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)
        char_rep_seq = []
        for i, word in enumerate(sentence):
            #print(i)
            #print(autograd.Variable(torch.LongTensor(targets.data[i])))
            model_char.zero_grad()
            char_seq_in = prepare_char(word, char_to_ix)
            char_rep, hidden = model_char(char_seq_in)
            char_rep, hidden = char_rep, hidden[0].view(-1).data.tolist()
            #print(len(hidden))
            #print(char_rep)
            char_rep_seq.append(hidden)
            target = autograd.Variable(torch.LongTensor
                                       ([targets.data[i] for j in range(len(word))]))
            loss = loss_function(char_rep, target)
            optimizer_char.step()

        #sequence_in = torch.cat([sentence_in,  char_rep_seq
        # Step 3. Run our forward pass.
        #print(char_rep_seq)
        char_in = torch.FloatTensor(char_rep_seq)
        char_in = autograd.Variable(char_in)
        
        tag_scores = model(sentence_in, char_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        
        loss = loss_function(tag_scores, targets)
        print(epoch, loss.data[0])
        loss.backward()
        optimizer.step()

# See what the scores are after training
inputs = prepare_sequence(training_data[0][0], word_to_ix)
char_rep_seq = []
for i, word in enumerate(training_data[0][0]):
    char_seq_in = prepare_char(word, char_to_ix)
    char_rep, hidden = model_char(char_seq_in)
    char_rep_seq.append(hidden[0].view(-1).data.tolist())


char_in = torch.FloatTensor(char_rep_seq)
char_in = autograd.Variable(char_in)
    
tag_scores = model(inputs, char_in)

print(tag_scores)
print(tag_to_ix)


