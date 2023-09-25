import subprocess
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import platform
import random
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import StandardScaler
from metrics import add_metrics
from TimesSeriesGenerator import TimeSeriesGenerator

subprocess.check_call(["pip", "install", "-r", "requirements.txt"])


# function to visualize input data
def show_raw_visualization(data):
    if df.shape[1] % 2 != 0:
        rows = (df.shape[1] + 1) / 2
    else:
        rows = df.shape[1] / 2
    rows = int(rows)
    fig, axes = plt.subplots(nrows=rows, ncols=2, figsize=(15, 18), dpi=80)
    for i in range(len(data.columns)):
        column = data.columns[i]
        color = (random.random(), random.random(), random.random())
        ax = data[column].plot(
            ax=axes[i // 2, i % 2],
            color=color,
            title='',
            rot=25,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.legend().remove()
        ax.set_title(column, loc="left", fontsize=12)
    plt.tight_layout()


# function to visualized correlation map
def show_heatmap(data):
    plt.figure(figsize=(11, 10))
    plt.matshow(data.corr(), fignum=2, vmin=-1, vmax=1, cmap="coolwarm")
    plt.xticks(range(data.shape[1]), data.columns, fontsize=8, rotation=90)
    plt.gca().xaxis.tick_bottom()
    plt.yticks(range(data.shape[1]), data.columns, fontsize=8)

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    plt.title("Data Correlation Heatmap", fontsize=14)


# function for picking elements with correlation at least 0.4

print("Python version:", platform.python_version())
print("TensorFlow version:", tf.version.VERSION)

print("Available GPU: ", len(tf.config.experimental.list_physical_devices('GPU')))

print("Check https://www.tensorflow.org/install/source#gpu to see software compatibility.")
print("Tensorflow up to 2.10.0 has support for GPU calculations on Windows")

# loading input data

file_name = 'pkn_d'
df = pd.read_csv(f'{file_name}.csv', index_col='Data', parse_dates=True)

df.info()

# adding metrics and visualization

df = add_metrics(df)

show_raw_visualization(df)

show_heatmap(df)

# Assuming `df` is your DataFrame
result = seasonal_decompose(df['Close'], model='additive', period=365, extrapolate_trend=0)

# Plot observed, trend, seasonal, and residuals
plt.figure(figsize=(12, 3), dpi=80)

plt.subplot(141)
plt.plot(result.observed, color='blue')
plt.title('Observed')

plt.subplot(142)
plt.plot(result.trend, color='green')
plt.title('Trend')

plt.subplot(143)
plt.plot(result.seasonal, color='red')
plt.title('Seasonal')

plt.subplot(144)
plt.plot(result.resid, color='purple')
plt.title('Residuals')

plt.tight_layout()

# display all the plots

plt.show()

# way to control the learning, validation and testing data sets; amount of epochs, nodes, name and location of the model
# what to predict set as "price_demo" will perform prediction on the closing price only
# such predictions are not considered to be professional as in financial prediction, returns are more important
# this was made only for visualisation purpose

what_to_predict = "log_return"
a = 80
b = 10
n_input = 7
y_output = 1
LSTM_nodes = 150
epochs = 15
comment = f'{a},{b},{100 - a}'
modelname = f'{file_name}_{n_input}DaysInput{y_output}DaysPrediction{LSTM_nodes}nodes{what_to_predict}'
folderpath = f'Modele/{modelname}/'

# calculation of data sets
train_size = int(len(df) * a / 100)
val_size = int(len(df) * b / 100)
test_size = len(df) - train_size - val_size

# dividing input data into data sets
train_data = df[:train_size]
val_data = df[train_size:train_size + val_size]
test_data = df[train_size + val_size:]

# writing down the size of each data set
print("Train set size:", len(train_data))
print("Validation set size:", len(val_data))
print("Test set size:", len(test_data))

# transforming the data sets into time series data sets, here also data is split into inputs and outputs of neural
# network

if what_to_predict == "price_demo":
    # standarization
    scaler = StandardScaler()

    scaler.fit(df)

    scaled_train = scaler.transform(train_data)
    scaled_test = scaler.transform(test_data)
    scaled_val = scaler.transform(val_data)

    n_features = 1
    generator_train = TimeSeriesGenerator(scaled_train[:, 0], scaled_train[:, 0], length=n_input,
                                          predict=y_output, mode='train')
    generator_val = TimeSeriesGenerator(scaled_val[:, 0], scaled_val[:, 0], length=n_input,
                                        predict=y_output, mode='train')
    generator_test = TimeSeriesGenerator(scaled_test[:, 0], scaled_test[:, 0], length=n_input,
                                         predict=y_output, mode='train')
else:
    # scaling
    scaler = StandardScaler()

    scaler.fit(df)

    scaled_train = scaler.transform(train_data)
    scaled_test = scaler.transform(test_data)
    scaled_val = scaler.transform(val_data)

    n_features = df.shape[1] - 1
    generator_train = TimeSeriesGenerator(np.delete(scaled_train, 0, axis=1), scaled_train[:, 2], length=n_input,
                                          predict=y_output, mode='train')
    generator_val = TimeSeriesGenerator(np.delete(scaled_val, 0, axis=1), scaled_val[:, 2], length=n_input,
                                        predict=y_output, mode='train')
    generator_test = TimeSeriesGenerator(np.delete(scaled_test, 0, axis=1), scaled_test[:, 2], length=n_input,
                                         predict=y_output, mode='train')


# loop used to check the resuling time series data sets with the use of debugging

for i in range(len(generator_train)):
    x, y = generator_train[i]
    print(x.shape, y.shape)

# setting for early stopping, and checkpoints

# with patience equal to epochs there will be no early stop essentially

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_root_mean_squared_error', verbose=1, patience=50)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_root_mean_squared_error', factor=0.005, patience=2,
                                                 min_lr=0.0001)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(f'{folderpath}{modelname}.h5',
                                                      monitor='val_root_mean_squared_error', save_best_only=True,
                                                      mode='min', verbose=1)
callbacks = [reduce_lr, model_checkpoint, early_stop]

# neural network structures

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(units=LSTM_nodes, input_shape=(n_input, n_features), dropout=0.2,
                         activation='tanh', kernel_initializer='glorot_normal', return_sequences=True),
    tf.keras.layers.LSTM(units=int(LSTM_nodes * 1.5), dropout=0.2,
                         activation='tanh', kernel_initializer='glorot_normal', return_sequences=False),
    tf.keras.layers.Dense(units=y_output, activation='LeakyReLU', kernel_initializer='he_normal')
])

# model.build(input_shape=(None, n_input, n_features))

# optimizer

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

# compiling the model

model.compile(optimizer=optimizer, loss='mean_squared_error',
              metrics=['accuracy', 'mean_absolute_error', 'RootMeanSquaredError'])

model.summary()

# training the model and creating the history

history = model.fit(generator_train, validation_data=generator_val, epochs=epochs, batch_size=1,
                    callbacks=callbacks, verbose=1, shuffle=True)

# saving the final model not necessary if using callbacks

model.save(f'{folderpath}{modelname}.h5')

# graphs for the output
# loss is mean squared error

loss_per_epoch = model.history.history['loss']
plt.plot(range(len(loss_per_epoch)), loss_per_epoch)
plt.savefig(f'{folderpath}Loss per epoch.png')
plt.show()

accuracy_per_epoch = model.history.history['accuracy']
plt.plot(range(len(accuracy_per_epoch)), accuracy_per_epoch)
plt.savefig(f'{folderpath}Accuracy per epoch.png')
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig(f'{folderpath}Model Accuracy.png')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig(f'{folderpath}Model Loss.png')
plt.show()

plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('Model mean absolute error')
plt.ylabel('Mean Absolute Error')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig(f'{folderpath}Model mean absolute error.png')
plt.show()

plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('Model root mean squared error')
plt.ylabel('Root mean squared error')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig(f'{folderpath}Model root mean squared error.png')
plt.show()

# testing of the new model with the use of test data set

test_loss, test_acc, test_mean_absolute_error, test_root_mean_square_error = model.evaluate(generator_test, verbose=1)
print("Test accuracy: {:.2f}%".format(test_acc * 100))
print("Test MSE: {:.2f}".format(test_loss))
print("Test MAE: {:.2f}".format(test_mean_absolute_error))
print("Test RMSE: {:.2f}".format(test_root_mean_square_error))
