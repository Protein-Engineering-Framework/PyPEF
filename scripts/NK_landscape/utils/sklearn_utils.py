import numpy as np
import pandas as pd

def csvDataLoader(csvfile,x_data="Sequence",y_data="Fitness",index_col=None):
    """Simple helper function to load NK landscape data from CSV files into numpy arrays.
    Supply outputs to sklearn_split to tokenise and split into train/test split.

    Parameters
    ----------

    csvfile : str

        Path to CSV file that will be loaded

    x_data : str, default="Sequence"

        String key used to extract relevant x_data column from pandas dataframe of
        imported csv file

    y_data : str, default="Fitness"

        String key used to extract relevant y_data column from pandas dataframe  of
        imported csv file

    index_col : int, default=None

        Interger value, if provided, will determine the column to use as the index column

    returns np.array (Nx2), where N is the number of rows in the csv file

        Returns an Nx2 array with the first column being x_data (sequences), and the second being
        y_data (fitnesses)
    """

    data      = pd.read_csv(csvfile,index_col=index_col)
    sequences = data[x_data].to_numpy()
    fitnesses = data[y_data].to_numpy()
    fitnesses = fitnesses.reshape(fitnesses.shape[0],1)
    sequences = sequences.reshape(sequences.shape[0],1)

    return np.concatenate((sequences, fitnesses), axis=1)

def collapse_concat(arrays,dim=0):
    """
    Takes an iterable of arrays and recursively concatenates them. Functions similarly
    to the reduce operation from python's functools library.

    Parameters
    ----------
    arrays : iterable(np.array)

        Arrays contains an iterable of np.arrays

    dim : int, default=0

        The dimension on which to concatenate the arrays.

    returns : np.array

        Returns a single np array representing the concatenation of all arrays
        provided.
    """
    if len(arrays) == 1:
        return arrays[0]
    else:
        return np.concatenate((arrays[0],collapse_concat(arrays[1:])))

def sklearn_tokenize(seqs, AAs='ACDEFGHIKLMNPQRSTVWY'):
    """
    Takes an iterable of sequences provided as one amino acid strings and returns
    an array of their tokenized form.

    TODO: Vectorize the operation

    Parameters
    ----------
    seqs : iterable of strings

        Iterable containing all strings

    AAs  : str, default="ACDEFGHIKLMNPQRSTVWY"

        The alphabet of permitted characters to tokenize. Single amino acid codes
        for all 20 naturally occurring amino acids is provided as default.

    returns : np.array(tokenized_seqs)

        Returns a tokenized NP array for the sequences.
    """
    tokens = {x:y for x,y in zip(AAs, list(range(len(AAs))))}
    return np.array([[tokens[aa] for aa in seq] for seq in seqs])


def sklearn_split(data, split=0.8):
    """
    Takes a dataset array of two layers, sequences as the [:,0] dimension and fitnesses
    as the [:,1] dimension, shuffles, and returns the tokenized sequences arrays
    and retyped fitness arraysself.

    Parameters
    ----------
    data : np.array (N x 2)

        The sequence and fitness data with sequences provided as single amino acid strings

    split : float, default=0.8, range (0-1)

        The split point for the training - validation data

    returns : x_train, y_train, x_test, y_test

        All Nx1 arrays with train as the first 80% of the shuffled data and test
        as the latter 20% of the shuffled data.
    """

    assert (0 < split < 1), "Split must be between 0 and 1"

    np.random.shuffle(data)

    split_point = int(len(data)*split)

    train = data[:split_point]
    test  = data[split_point:]

    x_train = train[:,0]
    y_train = train[:,1]
    x_test  = test[:,0]
    y_test  = test[:,1]

    return sklearn_tokenize(x_train).astype("int"), y_train.astype("float"), \
           sklearn_tokenize(x_test).astype("int"), y_test.astype("float")

def reset_params_skorch(regressor):
    """
    Simple helper function that manually resets the parameters in each layer of a
    skorch regressor model.

    Parameters
    ----------
    regressor : skorch.NeuralNetRegressor

        The neural net regressor (wrapped PyTorch model) that parameters must be
        reset (in place) for.

    returns  : None
    """
    regressor.get_params()["module"].zero_grad()
    for layer in regressor.get_params()["module"].children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
    return None

def train_test_model(model,x_train,y_train,x_test,y_test,return_model=False):
    """
    Takes a Sklearn (or Skorch) model, and both train and test data and fits, then
    scores the model on the data. Reshapes the y data if the model is a skorch
    wrapped system as it requires thatself.

    Parameters
    ----------
    model : sklearn model or skorch.NeuralNetRegressor

        The machine learning model that will be trained and then evaluted.

    x_train, y_train, x_test, y_test : np.array

        Data to train and test on. Must all be of the same length.

    return_model : Bool

        Boolean value that determines if the model will be returned.
    """

    print("Training model {} on {} data points".format(model,len(x_train)))


    if model.__class__.__name__ == "NeuralNetRegressor":
        reset_params_skorch(model)
        model.fit(x_train,y_train.reshape(-1,1))
        perf = model.score(x_test,y_test.reshape(-1,1))

    else:
        model.fit(x_train,y_train)
        perf = model.score(x_test,y_test)

    if return_model:
        return perf, model
    else:
        return perf

def save_landscape_dict(landscape_dict):
    """
    Simple helper function that takes a landscape dictionary and saves all of the
    landscapes within it.

    Parameters
    ----------
    landscape_dict : dict

        Dictionary of landscapes in format: {"name" : [Landscape_1,Landscape_2,..etc]}
    """
    for key in landscape_dict.keys():
        for landscape in landscape_dict[key]:
            landscape.save()

def load_dict(name):
    """
    Helper function that loads a given dictionary
    """
    with open(name, "rb") as file:
        return pkl.load(file)
