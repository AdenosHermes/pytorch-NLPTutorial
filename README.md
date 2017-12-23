# A Solution to the LSTM Exercise of the PyTorch NLP Tutorial 

The tutorial can be found at: <br>
http://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#sphx-glr-beginner-nlp-sequence-models-tutorial-py

This solution is a personal solution. It works, and works reasonably better than the original model. In the oridinal model, the loss goes down to 0.014 after 300 epochs, while this solution goes down to about 0.009.

To achieve better convergence, I added a 0.5 momentum to the SGD optimizer (to both the original model and my solution). Adding the momentum term alone improves the convergence rate of the original model by about 40%.

The solution is by no means the only solution or the optimal solution. I am not even sure whether I understood the intruction from the exercise correctly. But this is what I implemented:

1. A separate LSTM (called charLSTM) that takes in a the characters of a word, and outputs the category of the word.
2. The input to the original LSTM is now (sentence, hidden), where "hidden" is the concatenated hidden states of the charLSTM after iterating over the sentence, and I backpropagate charLSTM after every word, and the orginal LSTM after every sentence.

Personally, I think my solution is bad (let me know if you have any better solution).

# To Do
-Maybe add some plots to compare the results?
