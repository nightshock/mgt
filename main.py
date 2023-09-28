import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, SimpleRNN, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import keras_tuner as kt
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.layers import Flatten



# Загрузка данных
url = "https://min-api.cryptocompare.com/data/v2/histoday?fsym=BTC&tsym=USD&limit=1000"
response = requests.get(url)
data = response.json()['Data']['Data']
df = pd.DataFrame(data)
df['date'] = pd.to_datetime(df['time'], unit='s')
df = df[['date', 'close']]
df.set_index('date', inplace=True)

# Нормализуем данные
scaler = MinMaxScaler(feature_range=(0, 1))

# Сначала разделим данные
train_size = int(len(df) * 0.8)
train_data = df['close'][:train_size].values.reshape(-1, 1)
test_data = df['close'][train_size:].values.reshape(-1, 1)

# Теперь нормализуем обучающие данные
scaled_train_data = scaler.fit_transform(train_data)

# ... и используем этот же масштаб для тестовых данных
scaled_test_data = scaler.transform(test_data)

# 2. Оптимизация разделения данных
def create_dataset(data, look_back=60):
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

look_back = 60
X_train, y_train = create_dataset(scaled_train_data, look_back)
X_test, y_test = create_dataset(scaled_test_data, look_back)

# Reshape для соответствия формату модели
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Создаем датасет с последовательностями
X_data = []
y_data = []
scaled_data = np.concatenate((scaled_train_data, scaled_test_data))
for i in range(60, len(scaled_data)):
    X_data.append(scaled_data[i-60:i, 0])
    y_data.append(scaled_data[i, 0])

X_data, y_data = np.array(X_data), np.array(y_data)
X_data = np.reshape(X_data, (X_data.shape[0], X_data.shape[1], 1))

# Оценка стационарности временного ряда 
result = adfuller(df['close'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:')
for key, value in result[4].items():
    print(f'{key}: {value}')

# Добавим train-test-split для автоматического разделения
train_size = int(len(scaled_train_data) * 0.8)
X_train, X_temp, y_train, y_temp = train_test_split(X_data, y_data, train_size=train_size, shuffle=False)

# Разделение X_temp и y_temp на тестовый и валидационный наборы:
val_size = int(len(X_temp) * 0.5)  # 50% от оставшегося набора
X_val, X_test = X_temp[:val_size], X_temp[val_size:]
y_val, y_test = y_temp[:val_size], y_temp[val_size:]


def calculate_performance(df):
    # Реальные значения цен закрытия
    real_values = df['close'].values

    # Скользящее среднее
    moving_average = df['ma'].dropna().values

    # Расчет MSE между реальными значениями и скользящим средним
    mse = mean_squared_error(real_values[len(real_values) - len(moving_average):], moving_average)

    # Вычисление производительности (чем меньше MSE, тем лучше производительность)
    performance = -mse  # Минус MSE, так как мы хотим минимизировать ошибку

    return performance


# Создание скользящих средних
window_lengths = [7, 30, 90, 200, 365, 528]

for window_length in window_lengths:
    column_name = f'ma{window_length}'
    df[column_name] = df['close'].rolling(window=window_length).mean()

def build_model(hp):
    model = Sequential()
    
    model_type = hp.Choice('model_type', ['lstm', 'gru', 'simple_rnn', '1d_cnn'])
    
    # LSTM model
    if model_type == 'lstm':
        for i in range(hp.Int('num_lstm_layers', 1, 4)):
             model.add(LSTM(units=hp.Int('units_lstm_' + str(i), min_value=50, max_value=150, step=10),
                       return_sequences=True))
             model.add(Dropout(rate=hp.Float('dropout_lstm_' + str(i), 0.1, 0.5, step=0.1)))

    # GRU model
    elif model_type == 'gru':
        for i in range(hp.Int('num_gru_layers', 1, 4)):
            model.add(GRU(units=hp.Int('units_gru_' + str(i), min_value=50, max_value=150, step=10),
                      return_sequences=True))
            model.add(Dropout(rate=hp.Float('dropout_gru_' + str(i), 0.1, 0.5, step=0.1)))

    # SimpleRNN model
    elif model_type == 'simple_rnn':
        for i in range(hp.Int('num_rnn_layers', 1, 4)):
            model.add(SimpleRNN(units=hp.Int('units_rnn_' + str(i), min_value=50, max_value=150, step=10),
                          return_sequences=True))
            model.add(Dropout(rate=hp.Float('dropout_rnn_' + str(i), 0.1, 0.5, step=0.1)))

    # 1D-CNN model
    elif model_type == '1d_cnn':
        model.add(Conv1D(filters=hp.Int('filters', min_value=32, max_value=128, step=32), kernel_size=hp.Int('kernel_size', 2, 6, 2), activation='relu', input_shape=(X_data.shape[1], 1)))
        model.add(MaxPooling1D(pool_size=hp.Int('pool_size', 2, 4, 1)))
        model.add(Dropout(rate=hp.Float('dropout_cnn', 0.1, 0.5, step=0.1)))
        model.add(Flatten())

    model.add(Dense(units=1)) # This line should be here

    optimizer = hp.Choice('optimizer', ['adam', 'rmsprop'])
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model


tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=30,
    executions_per_trial=1,
    directory='keras_tuner_dir',
    project_name='bitcoin_forecasting'
)


# Добавляем ModelCheckpoint в callbacks
checkpoint_path = "model_weights.h5"
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min')
early_stop = EarlyStopping(monitor='val_loss', patience=10)
callbacks = [early_stop, checkpoint]

tuner.search(X_train, y_train, epochs=100, validation_split=0.2, callbacks=callbacks)
best_model = tuner.get_best_models(num_models=1)[0]

# Обучение лучшей модели с сохранением весов
history = best_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, callbacks=[checkpoint])


# Разделение данных на обучающий и тестовый наборы
train_data = scaled_train_data[:-200]
test_data = scaled_test_data[-260:]  # Включая 60 предыдущих для создания последовательности

# Создание X_test и y_test:
X_test = []
y_test = df['close'].iloc[-200:].values

for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i, 0])
    
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Прогнозирование с помощью модели:
predicted_values = best_model.predict(X_test)

# Исправляем форму массива и проводим обратное преобразование масштаба:
predicted_values_reshaped = predicted_values.squeeze().reshape(-1, 1)
predicted_values_unscaled = scaler.inverse_transform(predicted_values_reshaped)


# Оценка ошибки модели на обучающих данных
train_loss = best_model.evaluate(X_train, y_train, verbose=0)
print(y_train.shape)
print(best_model.predict(X_train).squeeze().shape)
train_mae = mean_absolute_error(y_train, best_model.predict(X_train).squeeze())
print(f"Ошибка MSE на обучающих данных: {train_loss:.4f}")
print(f"Ошибка MAE на обучающих данных: {train_mae:.4f}")

# Оценка ошибки модели на тестовых данных
test_loss = best_model.evaluate(X_test, y_test, verbose=0)
print(y_test.shape)
print(best_model.predict(X_test).squeeze().shape)
test_mae = mean_absolute_error(test_data[look_back:], predicted_values_unscaled)
print(f"Ошибка MAE на тестовых данных: {test_mae:.4f}")

# Оценка статистики данных
statistics = df['close'].describe()

# Вывод статистических показателей
print(statistics)

# Визуализация результатов с использованием Plotly:
fig = go.Figure()

fig.add_trace(go.Scatter(x=df.index[-200:], y=y_test, mode='lines', name='Реальные значения'))
fig.add_trace(go.Scatter(x=df.index[-200:], y=predicted_values.flatten(), mode='lines', name='Прогнозируемые значения'))

fig.update_layout(title='Прогнозирование цены закрытия акций', xaxis_title='Время', yaxis_title='Цена закрытия акций')
fig.show()

# Создание графика временного ряда
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Цены закрытия', line=dict(color='blue')))
fig.update_layout(title='Временной ряд цен закрытия биткоина', xaxis_title='Дата', yaxis_title='Цена закрытия')
fig.show()

# Создание новой модели и загрузка весов
new_model = build_model()  # создание нового экземпляра модели
new_model.load_weights('best_model.h5')

# Вывод результатов
print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:')
for key, value in result[4].items():
    print(f'{key}: {value}')