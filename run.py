import torch
from neur.model import Model
from neur.load_data import load_data
from neur.preprocess import train_test_split


def main():
    # running now on real data!
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.2)
    model = Model(data=(X_train, X_validate, X_test), labels=(y_train, y_validate, y_test))
    model.model_validation()

    # fix hyper parameters:
    hidden = model.size_hidden[get_hidden(model)]
    epochs = get_epochs()
    model.set_hyper_params(hidden=hidden, epochs=epochs)

    # train final model


def get_hidden(model):
    hidden = int(input(f"What size of hidden vector should be used?\n{model.size_hidden}\nType index of value:"))
    if not 0 <= hidden < model.size_hidden.shape[0]:
        raise Warning(f'Careful! Size should be integer in [0,{model.size_hidden.shape[0]})')
        hidden = int(input(f"What size of hidden vector should be used?\n{model.size_hidden}\nType index of value:"))
        if not 0 <= hidden < model.size_hidden.shape[0]:
            raise Exception("Incorrect hidden size!")
    return hidden


def get_epochs():
    epochs = int(input(f"What ammount of epochs should be used?\nType value:"))
    if not 0 < epochs:
        raise Warning(f'Careful! Size should be integer greater than 0')
        epochs = int(input(f"What ammount of epochs should be used?\nType value:"))
        if not 0 < epochs:
            raise Exception("Incorrect epochs!")
    return epochs


if __name__ == '__main__':
    main()
