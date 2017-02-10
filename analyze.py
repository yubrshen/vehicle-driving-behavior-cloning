import numpy as np

from astropy.visualization import hist
#from data import record_list
import matplotlib.pyplot as plt

# steering_angles = [float(r['steering']) for r in record_list]
# speeds = [float(r['speed']) for r in record_list]
# hist(steering_angles, bins="blocks")

# hist(speeds, bins="blocks")

# speed =  [float(r['speed']) for r in record_list]
# throttle = [float(r['throttle']) for r in record_list]
# plt.plot(throttle, speed, 'b.')

def plot_images_with_steering(x, y):
    import matplotlib.gridspec as gridspec
    img_num = x.shape[0]
    gs1 = gridspec.GridSpec(1, img_num)
    gs1.update(wspace=0.9, hspace=0.9) # set the spacing between axes. 
    plt.figure(figsize=(img_num*4, 2))
    for i in range(img_num):
        ax1 = plt.subplot(gs1[i])
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect('equal')
        ax1.set_title(str.format('steering: {}', y[i] ))
        plt.imshow(x[i])
    plt.show()

# i = 0
# while True:
#     augumented = augument_samples(record_list[i])
#     i = i+1
#     if augumented:
#         break
# x, y = augumented
# plot_images_with_steering(np.array(x), np.array(y))

# find sample photo with steering in interval:
def sample_with_steering_in_interval(records, a=0, b=1):
    i = 0
    for i in range(len(records)):
        if a < abs(float(records[i]['steering'])) < b:
            break
    images = [mpimg.imread(os.path.join(records[i]['dir'],
                                        records[i][camera].strip())) for camera in ['left', 'center', 'right']]
    steerings = ["", float(records[i]['steering']), ""]
    analyze.plot_images_with_steering(np.array(images), steerings)
    return records[i]

# Show the trajectory of the training history
def show_history(history):
    plt.plot(history.history['val_loss'], 'b-', label='validation')
    plt.plot(history.history['loss'], 'g-', label = 'training')
    plt.legend(mode='best')
    return history

# Show the predictions compared with the targets
def show_predictions(model, generator, number_to_show=None, label='training', threshold=0.1):
    x, y = next(generator)
    number_to_show = min(y.shape[0], number_to_show or y.shape[0])
    predictions = model.predict_on_batch(x)
    predictions = predictions[:, 0]

    delta = predictions - y

    ok_predictions = np.zeros(number_to_show)
    bad_predictions = np.zeros(number_to_show)
    
    ok_indices = abs(delta) < threshold
    ok_predictions[ok_indices] = predictions[ok_indices]

    bad_indices = threshold <= abs(delta)
    bad_predictions[bad_indices] = predictions[bad_indices]

    print('The maximum in magnitude of the delta: ', max(abs(delta)))
    bar_width = 0.2
    plt.figure(figsize=(15, number_to_show*bar_width))
    signal_seq = range(number_to_show)
    plt.barh(signal_seq, y[:number_to_show], align='center', alpha=0.4, label='target', color='black', hatch='|', edgecolor = 'black')
    plt.barh(signal_seq, ok_predictions, align='center', alpha=0.4, label='OK prediction', color='yellow')
    plt.barh(signal_seq, bad_predictions, align='center', alpha=0.4, label='Bad prediction', color='red')
    plt.legend(loc='best')
    plt.title('Prediction performance of ' + label)
    return model
