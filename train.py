import numpy as np

def shuffle_weights(model, weights=None):
    """Randomly permute the weights in `model`, or the given `weights`.
    This is a fast approximation of re-initializing the weights of a model.
    Assumes weights are distributed independently of the dimensions of the weight tensors
      (i.e., the weights have the same distribution along each dimension).
    :param Model model: Modify the weights of the given model.
    :param list(ndarray) weights: The model's weights will be replaced
     by a random permutation of these weights.
      If `None`, permute the model's current weights.
    """
    if weights is None:
        weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    model.set_weights(weights)

from data import gen_samples, shuffle_balance_split
import math

def train(model, data_sources, batch_size=64, epochs=29, verbose=0,
          split=10, steering_keep_percentage=0.1):
    """
    train with model, data source, and exiting weights, if existing.

    Returns the training history, and the trained model
    """
    train_list, validation_list, dropped = shuffle_balance_split(
        data_sources, split=split,
        steering_keep_percentage=steering_keep_percentage)
    # use 10 percent of the data for validation
    # will review if I should use those dropped samples of small steering angles.
    # Thay may be good for pure augmentation on the left, and right cameras emulation.

    from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
    history = model.fit_generator(gen_samples(train_list, batch_size=batch_size, augment=True),
                                  samples_per_epoch=len(train_list),
                                  nb_epoch=epochs, verbose=verbose,
                                  validation_data=gen_samples(
                                      validation_list, batch_size=batch_size, augment=False),
                                  nb_val_samples=len(validation_list),
                                  callbacks=[
                                      ModelCheckpoint(
                                          filepath="./model-current/model-{epoch:02d}-{val_loss:.2f}.h5",
                                          verbose=1, save_best_only=True, period=30)])
    print('Training ends...')
    return history, model

def get_input_with_default(prompt_name='', default_val=None):
    print('{}? The default is {}'.format(prompt_name, default_val))
    return input() or default_val

if __name__ == '__main__':
    from model import steering_model
    from keras.optimizers import Adam
    from keras.models import load_model
    # prompt for model path
    model_path = get_input_with_default('File path for the model')
    if model_path:
        model = load_model(model_path) 
    else:
        lr = float(get_input_with_default('The new learning rate', 0.0001))
        model = steering_model(optimizer=Adam(lr=lr), loss='mse', metrics=['accuracy'])
    model.summary()
    sample_dir = get_input_with_default('Directory for samples', './data_from_udacity/')
    print('Training starts, please take a break!')
    history, model = train(model, [sample_dir], batch_size=128, epochs=29,
                           steering_keep_percentage=0.1, verbose=0)
    # TODO: analyze the accuracy
    model_path = get_input_with_default('File path to save the model definition',
                                        model_path or './model-current/model.h5')
    model.save(model_path)
