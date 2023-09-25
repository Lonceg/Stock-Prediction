import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from metrics import add_metrics

file_name = 'pkn_d'
df = pd.read_csv(f'{file_name}.csv', header=0)

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

Daty = pd.DataFrame({'Daty': df['Data'].tail(n_input).values})
new_dates = pd.date_range(start=Daty['Daty'].iloc[-1], periods=y_output+1, freq='D')[1:]
new_dates = pd.to_datetime(new_dates)
Daty['Daty'] = pd.to_datetime(Daty['Daty'])
Daty['Daty'] = Daty['Daty'].dt.date
df = df.drop(['Data'], axis=1)
df = add_metrics(df)

model = tf.keras.models.load_model(f'{folderpath}{modelname}.h5')

if what_to_predict == "price_demo":
    scaler = StandardScaler()
    resultsscaler = StandardScaler()

    results = df.values
    input_data = df[-n_input:]

    n_features = 1
    results = results[:, 0]
    results = results.reshape(-1, 1)
    scaler.fit(df)
    resultsscaler.fit(results)
    input_data_scaled = scaler.transform(input_data)
    input_data_scaled = input_data_scaled[:, 0]
    input_data = input_data['Close'].values
else:
    scaler = StandardScaler()
    resultsscaler = StandardScaler()

    results = df.values
    input_data = df[-n_input:]

    n_features = df.shape[1] - 1
    results = results[:, 2]
    results = results.reshape(-1, 1)
    scaler.fit(df)
    resultsscaler.fit(results)
    input_data_scaled = scaler.transform(input_data)
    input_data_scaled = input_data_scaled[:, 2]
    input_data = input_data['Log Return'].values

input_data_scaled = np.expand_dims(input_data_scaled, axis=0)
predictions = model.predict(input_data_scaled)
predictions = resultsscaler.inverse_transform(predictions)

plt.plot(Daty['Daty'], input_data, label=f'Input data {n_input} days', color='blue')
plt.plot(new_dates, predictions[0], label=f'Prediction {y_output} days', color='orange')

for date in new_dates:
    plt.axvline(date, color='black', linestyle='--', alpha=0.5)

plt.title('Plot of the stock prediction')
plt.xlabel('Dates')
plt.ylabel('Values')
plt.xticks(rotation=60)
plt.legend()
plt.tight_layout()
plt.savefig(f'{folderpath}Plot of the stock prediction.png')
plt.show()

