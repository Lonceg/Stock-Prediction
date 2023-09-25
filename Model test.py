import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from metrics import add_metrics
from TimesSeriesGenerator import TimeSeriesGenerator


def convert_generator_to_numpy(source):
    data, targets = [], []
    for i in range(len(source)):
        x, y = source[i]
        data.append(x)
        targets.append(y)

    data = np.vstack(data)
    targets = np.vstack(targets)
    return data, targets

file_name = 'pkn_d'
df = pd.read_csv(f'{file_name}.csv', index_col='Data', parse_dates=True)

df = add_metrics(df)

what_to_predict = "price_demo"
a = 80
b = 10
n_input = 30
y_output = 30
LSTM_nodes = 150
epochs = 15
comment = f'{a},{b},{100 - a}'
modelname = f'{file_name}_{n_input}DaysInput{y_output}DaysPrediction{LSTM_nodes}nodes{what_to_predict}'
folderpath = f'Modele/{modelname}/'

model = tf.keras.models.load_model(f'{folderpath}{modelname}.h5')

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



if what_to_predict == "price_demo":
    scaler = StandardScaler()
    resultsscaler = StandardScaler()

    results = df.values

    n_features = 1
    results = results[:, 0]
    results = results.reshape(-1, 1)
    scaler.fit(df)
    resultsscaler.fit(results)
    scaled_train = scaler.transform(train_data)
    scaled_train_data_results = resultsscaler.transform(train_data.values[:, 0].reshape(-1, 1))
    scaled_test = scaler.transform(test_data)
    scaled_test_data_results = resultsscaler.transform(test_data.values[:, 0].reshape(-1, 1))
    generator_train = TimeSeriesGenerator(scaled_train[:, 0], scaled_train[:, 0], length=n_input,
                                          predict=y_output, mode='test')
    generator_test = TimeSeriesGenerator(scaled_test[:, 0], scaled_test[:, 0], length=n_input,
                                         predict=y_output, mode='test')

    average_value = df['Close'].mean()

else:
    scaler = StandardScaler()
    resultsscaler = StandardScaler()

    results = df.values

    n_features = df.shape[1] - 1
    results = results[:, 2]
    results = results.reshape(-1, 1)
    scaler.fit(df)
    resultsscaler.fit(results)
    scaled_train = scaler.transform(train_data)
    scaled_train_data_results = resultsscaler.transform(train_data.values[:, 2].reshape(-1, 1))
    scaled_test = scaler.transform(test_data)
    scaled_test_data_results = resultsscaler.transform(test_data.values[:, 2].reshape(-1, 1))
    generator_train = TimeSeriesGenerator(np.delete(scaled_train, 0, axis=1), scaled_train[:, 2], length=n_input,
                                          predict=y_output, mode='test')
    generator_test = TimeSeriesGenerator(np.delete(scaled_test, 0, axis=1), scaled_test[:, 2], length=n_input,
                                         predict=y_output, mode='test')

    average_value = df['Log Return'].mean()

for i in range(len(generator_test)):
    x, y = generator_test[i]
    print(x.shape, y.shape)

numpytraindata, numpytraintargets = convert_generator_to_numpy(generator_train)
numpytestdata, numpytesttargets = convert_generator_to_numpy(generator_test)

test_loss, test_acc, test_mean_absolute_error, test_root_mean_square_error = model.evaluate(numpytestdata,
                                                                                            numpytesttargets, verbose=1)
print("Test accuracy: {:.2f}%".format(test_acc * 100))

rescaledMSE = resultsscaler.inverse_transform(np.reshape(test_loss, (1, -1)))[0][0]
rescaledMAE = resultsscaler.inverse_transform(np.reshape(test_mean_absolute_error, (1, -1)))[0][0]
rescaledRMSE = resultsscaler.inverse_transform(np.reshape(test_root_mean_square_error, (1, -1)))[0][0]
print("Scaled MSE (batch-wise calculation of fit()): {:.2f}".format(test_loss))
print("Scaled MAE: {:.2f}".format(test_mean_absolute_error))
print("Scaled RMSE: {:.2f}".format(test_root_mean_square_error))
print("Rescaled MSE: (batch-wise calculation of fit()): {:.2f}".format(rescaledMSE))
print("Rescaled MAE: {:.2f}".format(rescaledMAE))
print("Rescaled RMSE: {:.2f}".format(rescaledRMSE))
print("Rescaled RMSE as a percentage of average: {:.2f}%".format((rescaledRMSE / average_value) * 100))

y_actual_train = numpytraintargets
y_actual_test = numpytesttargets

y_pred_train = model.predict(numpytraindata)
y_pred_test = model.predict(numpytestdata)

timesteps = y_pred_train.shape[1]
sequences = y_pred_train.shape[0]
y_pred_train_reshaped = y_pred_train.reshape(sequences * timesteps, -1)

y_pred_train_reshaped = resultsscaler.inverse_transform(y_pred_train_reshaped)

y_actual_train_reshaped = y_actual_train.reshape(sequences * timesteps, -1)

y_actual_train_reshaped = resultsscaler.inverse_transform(y_actual_train_reshaped)

timesteps = y_pred_test.shape[1]
sequences = y_pred_test.shape[0]

y_pred_test_reshaped = y_pred_test.reshape(sequences * timesteps, -1)

y_pred_test_reshaped = resultsscaler.inverse_transform(y_pred_test_reshaped)

y_actual_test_reshaped = y_actual_test.reshape(sequences * timesteps, -1)

y_actual_test_reshaped = resultsscaler.inverse_transform(y_actual_test_reshaped)

mse = np.mean((y_actual_train_reshaped - y_pred_train_reshaped) ** 2)
mae = np.mean(np.abs(y_actual_train_reshaped - y_pred_train_reshaped))
rmse = np.sqrt(np.mean((y_actual_train_reshaped - y_pred_train_reshaped) ** 2))

mse_percent = (mse / average_value ** 2) * 100
mae_percent = (mae / average_value) * 100
rmse_percent = (rmse / average_value) * 100

errors = y_pred_train_reshaped - y_actual_train_reshaped
std_dev = np.std(errors)
y_pred_train_reshaped = y_pred_train_reshaped[:, 0]

print("Calculated train MSE: {:.2f}".format(mse))
print("Calculated train MAE: {:.2f}".format(mae))
print("Calculated train RMSE: {:.2f}".format(rmse))
print("Calculated train MSE as a percentage of average: {:.2f}%".format(mse_percent))
print("Calculated train MAE as a percentage of average: {:.2f}%".format(mae_percent))
print("Calculated train RMSE as a percentage of average: {:.2f}%".format(rmse_percent))

df = pd.DataFrame(y_pred_train_reshaped)
df.to_csv('Prediction_train.csv', index=False)

plt.plot(y_actual_train_reshaped, label='Actual')
plt.plot(y_pred_train_reshaped, label='Predicted')
plt.legend()
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Actual vs Predicted Values Train')
plt.savefig(f'{folderpath}Actual vs Predicted Values Train.png')
plt.show()

mse = np.mean((y_actual_test_reshaped - y_pred_test_reshaped) ** 2)
mae = np.mean(np.abs(y_actual_test_reshaped - y_pred_test_reshaped))
rmse = np.sqrt(np.mean((y_actual_test_reshaped - y_pred_test_reshaped) ** 2))
mse_percent = (mse / average_value ** 2) * 100
mae_percent = (mae / average_value) * 100
rmse_percent = (rmse / average_value) * 100

errors = y_pred_test_reshaped - y_actual_test_reshaped
y_pred_test_reshaped = y_pred_test_reshaped[:, 0]

print("Calculated test MSE: {:.2f}".format(mse))
print("Calculated test MAE: {:.2f}".format(mae))
print("Calculated test RMSE: {:.2f}".format(rmse))
print("Calculated test MSE as a percentage of average: {:.2f}%".format(mse_percent))
print("Calculated test MAE as a percentage of average: {:.2f}%".format(mae_percent))
print("Calculated test RMSE as a percentage of average: {:.2f}%".format(rmse_percent))

df = pd.DataFrame(y_pred_test_reshaped)
df.to_csv('Prediction_test.csv', index=False)

plt.plot(y_actual_test_reshaped, label='Actual')
plt.plot(y_pred_test_reshaped, label='Predicted')
plt.legend()
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Actual vs Predicted Values Test')
plt.savefig(f'{folderpath}Actual vs Predicted Values Test.png')
plt.show()
