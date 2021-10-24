# TV-Script-Generation---RNN
This project is part of Udacity nanodegree on Deep Learning. The objective is to generate own Seinfeld TV scripts using Recurrent Neural Networks (RNNs)

You'll be using part of the Seinfeld dataset of scripts from 9 seasons. The Neural Network you'll build will generate a new ,"fake" TV script, based on patterns it recognizes in this training data.

### Get the Data
The data is already provided for you in ./data/Seinfeld_Scripts.txt


## Implement Pre-processing Functions
The first thing to do to any dataset is pre-processing. We implement the following pre-processing functions as below:

- Lookup Table
- Tokenize Punctuation

### Lookup Table
To create a word embedding, you first need to transform the words to ids. In this function, create two dictionaries:
 - Dictionary to go from the words to an id, we'll call vocab_to_int
 - Dictionary to go from the id to word, we'll call int_to_vocab
Return these dictionaries in the following tuple (vocab_to_int, int_to_vocab)


## Tokenize Punctuation
We'll be splitting the script into a word array using spaces as delimiters. However, punctuations like periods and exclamation marks can create multiple ids for the same word. For example, "bye" and "bye!" would generate two different word ids.
Implement the function token_lookup to return a dict that will be used to tokenize symbols like "!" into "||Exclamation_Mark||". Create a dictionary for the following symbols where the symbol is the key and value is the token:
- Period ( . )
- Comma ( , )
- Quotation Mark ( " )
- Semicolon ( ; )
- Exclamation mark ( ! )
- Question mark ( ? )
- Left Parentheses ( ( )
- Right Parentheses ( ) )
- Dash ( - )
- Return ( \n )
This dictionary will be used to tokenize the symbols and add the delimiter (space) around it. This separates each symbols as its own word, making it easier for the neural network to predict the next word. Make sure you don't use a value that could be confused as a word; for example, instead of using the value "dash", try using something like "||dash||".

### Build the Neural Network
Here we build the components necessary to build an RNN by implementing the RNN Module and forward and backpropagation functions.

#### Input
We have used TensorDataset to provide a known format to our dataset; in combination with DataLoader, it will handle batching, shuffling, and other dataset iteration functions.

data = TensorDataset(feature_tensors, target_tensors)
data_loader = torch.utils.data.DataLoader(data, 
                                          batch_size=batch_size)
#### Batching
Implement the batch_data function to batch words data into chunks of size batch_size using the TensorDataset and DataLoader classes.
Batch words using the DataLoader, create feature_tensors and target_tensors of the correct size and content for a given sequence_length.

#### Sizes
Your sample_x should be of size (batch_size, sequence_length) or (10, 5) in this case and sample_y should just have one dimension: batch_size (10).

#### Values
You should also notice that the targets, sample_y, are the next value in the ordered test_text data. So, for an input sequence [ 28,  29,  30,  31,  32] that ends with the value 32, the corresponding output should be 33.

#### Build the Neural Network
Implement an RNN using PyTorch's Module class. You may choose to use a GRU or an LSTM. To complete the RNN, you'll have to implement the following functions for the class:

__init__ - The initialize function.
"""
        Initialize the PyTorch RNN Module
        :param vocab_size: The number of input dimensions of the neural network (the size of the vocabulary)
        :param output_size: The number of output dimensions of the neural network
        :param embedding_dim: The size of embeddings, should you choose to use them        
        :param hidden_dim: The size of the hidden layer outputs
        :param dropout: dropout to add in between LSTM/GRU layers
        """

init_hidden - The initialization function for an LSTM/GRU hidden state
'''
        Initialize the hidden state of an LSTM/GRU
        :param batch_size: The batch_size of the hidden state
        :return: hidden state of dims (n_layers, batch_size, hidden_dim)
        '''

forward - Forward propagation function.
"""
        Forward propagation of the neural network
        :param nn_input: The input to the neural network
        :param hidden: The hidden state        
        :return: Two Tensors, the output of the neural network and the latest hidden state
        """

The initialize function should create the layers of the neural network and save them to the class. The forward propagation function will use these layers to run forward propagation and generate an output and a hidden state.

The output of this model should be the last batch of word scores after a complete sequence has been processed. That is, for each input sequence of words, we only want to output the word scores for a single, most likely, next word.

## Define forward and backpropagation
Use the RNN class you implemented to apply forward and back propagation. This function will be called, iteratively, in the training loop as follows:

loss = forward_back_prop(decoder, decoder_optimizer, criterion, inp, target)
And it should return the average loss over a batch and the hidden state returned by a call to RNN(inp, hidden). Recall that you can get this loss by computing it, as usual, and calling loss.item().

### Neural Network Training
With the structure of the network complete and data ready to be fed in the neural network, it's time to train it.

### Train Loop
The training loop is implemented for you in the train_decoder function. This function will train the network over all the batches for the number of epochs given. The model progress will be shown every number of batches. This number is set with the show_every_n_batches parameter. You'll set this parameter along with other parameters in the next section

Hyperparameters
Set and train the neural network with the following parameters:

Set sequence_length to the length of a sequence.
Set batch_size to the batch size.
Set num_epochs to the number of epochs to train for.
Set learning_rate to the learning rate for an Adam optimizer.
Set vocab_size to the number of uniqe tokens in our vocabulary.
Set output_size to the desired size of the output.
Set embedding_dim to the embedding dimension; smaller than the vocab_size.
Set hidden_dim to the hidden dimension of your RNN.
Set n_layers to the number of layers/cells in your RNN.
Set show_every_n_batches to the number of batches at which the neural network should print progress.
If the network isn't getting the desired results, tweak these parameters and/or the layers in the RNN class.

Train

I have experimented with different sequence lengths, which determine the size of the long range dependencies that a model can learn.

Results: 
sequence_length = 12 Sequence Length of twelve is slightly above an average sentence length of 6-8, however produces good results. batch_size = 128 Batch size in range 64 - 256 is recommneded. I have chosen 128 num_epochs = 20 Epochs seem sufficient to get desired loss below 3.5. More training might result in slightly lower loss. learning_rate = 0.001 Based on trial and error in 0.001-0.01 range this gives a good results. vocab_size = len(vocab_to_int) embedding_dim and hidden_dim = 256 based on previous excercises and lectures in the course n_layers = 2
