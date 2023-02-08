from neur.model import Model
from load_data import load_data
from neur.split import train_test_split
import warnings
import csv


def main():
    """ Full set up of network
    Load and splitting data, validate, train and assess model.
    """
    # Loading data
    X, y = load_data()
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.2)
    # Initialise model
    model = Model(data=(X_train, X_validate, X_test), labels=(y_train, y_validate, y_test))
    # Perform model validation over hyper parameters
    model.model_validation(epochs=40)

    # fix hyper parameters:
    final_hidden = model.size_hidden[get_hidden(model)]
    final_epochs = get_epochs()
    model.set_hyper_params(hidden=final_hidden, epochs=final_epochs)

    # train final model
    training_loss = model.train_model()
    # assess model
    test_error = model.assess()
    # log final error
    with open('logs/test_error.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['training_error', 'test_error'])
        writer.writerow([training_loss, test_error])
    print(f'Training error is {training_loss}\nTest error is {test_error}')
    return


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
    main()
