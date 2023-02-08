from neur.model import Model
from load_data import load_data
from neur.split import train_test_split
from setup_data import set_up_data
import warnings
import csv
import time
import sys


def print_custom(s='', silent=False):
    if silent:
        return
    print(s)


def main(silent=False):
    """ Full set up of network
    Load and splitting data, validate, train and assess and save model.
    :param silent: Whether show progress
    """
    # Loading data
    # Maximum length = 1396
    print_custom('loading data...', silent)
    X, y = load_data(time=1396)

    # Split data
    print_custom('splitting data...', silent)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.2)

    # Initialize model
    print_custom('initializing model...', silent)
    model = Model(data=(X_train, X_validate, X_test), labels=(y_train, y_validate, y_test))

    # Perform model validation over hyper parameters
    print_custom('perform model validation...', silent)
    model.model_validation(epochs=40)

    # fix hyper parameters:
    print_custom('set hyper parameters...', silent)
    final_hidden = model.size_hidden[get_hidden(model)]
    final_epochs = get_epochs()
    model.set_hyper_params(hidden=final_hidden, epochs=final_epochs)

    # train final model
    print_custom('train final model...', silent)
    training_loss = model.train_model()

    # assess model
    print_custom('assess model...', silent)
    test_error = assess_model(model=model, training_loss=training_loss)

    print_custom('workflow complete.', silent)
    model.save_model()
    print(f'Training error is {training_loss}\nTest error is {test_error}')
    return


def assess_model(model, training_loss):
    """
    Assess model after training has been performed.
    :param model: model
    :param training_loss: training loss for logging purposes
    :return: test error
    """
    # assess model
    test_error = model.assess()
    # log final error
    with open('logs/test_error.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['training_error', 'test_error'])
        writer.writerow([training_loss, test_error])
    return test_error


def full_workfow():
    """
    Running full learning workflow
    Preprocessing and validating, training and assessing model
    """
    set_up_data()
    main()


def get_hidden(model):
    hidden = int(input(f"What size of hidden vector should be used?\n{model.size_hidden}\nType index of value: "))
    if not 0 <= hidden < len(model.size_hidden):
        warnings.warn(f'Careful! Size should be integer in [0,{len(model.size_hidden)})')
        hidden = int(input(f"What size of hidden vector should be used?\n{model.size_hidden}\nType index of value: "))
        if not 0 <= hidden < len(model.size_hidden):
            raise Exception("Incorrect hidden size!")
    return hidden


def get_epochs():
    epochs = int(input(f"What amount of epochs should be used?\nType value: "))
    if not 0 < epochs:
        warnings.warn(f'Careful! Size should be integer greater than 0')
        epochs = int(input(f"What amount of epochs should be used?\nType value: "))
        if not 0 < epochs:
            raise Exception("Incorrect epochs!")
    return epochs


if __name__ == '__main__':
    startTime = time.time()
    globals()[sys.argv[1]]()
    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))
