import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import wandb
from wandb.keras import WandbCallback
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import functions as fn


wandb.login()

config = {
    "batch_size": 64,
    "epochs": 100,
    "timesteps": 7,
    "hidden_units": 50,
    "dropout_rate": 0.2,
    "learning_rate": 0.001
}

wandb.init(project="DM1", config=config)


def initialize_model(X_train, y_train):
    model = Sequential()
    model.add(LSTM(config["hidden_units"], input_shape=(
        config["timesteps"], X_train.shape[2])))
    model.add(Dropout(config["dropout_rate"]))
    # output layer
    model.add(Dense(3, activation="softmax"))

    optimizer = tf.keras.optimizers.Adam(learning_rate=config["learning_rate"])
    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizer, metrics=["accuracy"])

    return model


def create_sequences(data, timesteps, n_target_columns):
    X, y = [], []
    for i in range(len(data) - timesteps):
        X.append(data[i: i + timesteps, :-n_target_columns])
        y.append(data[i + timesteps, -n_target_columns:])
    return np.array(X), np.array(y)


def main():
    data = fn.clean_data()
    # Drop missing values
    data = data.dropna()

    # Split the dataset into input features and target
    X = data.drop(columns=['time', 'mood'])
    y = data['mood']

    # One-hot encode the target
    encoder = OneHotEncoder(sparse=False, categories="auto")
    y_encoded = encoder.fit_transform(y.values.reshape(-1, 1))

    # Concatenate X and y
    data_concat = np.hstack([X, y_encoded])
    # Create time series sequences
    n_target_columns = y_encoded.shape[1]
    X_sequences, y_sequences = create_sequences(
        data_concat, config["timesteps"], n_target_columns)


    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_sequences, y_sequences, test_size=0.2, random_state=42)
    model = initialize_model(X_train, y_train)

    history = model.fit(
        X_train,
        y_train,
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        validation_split=0.1,
        callbacks=[WandbCallback()],
    )

    model.evaluate(X_test, y_test)

    # save model plus timestamp
    model.save(f"models/{wandb.run.id}.h5")


if __name__ == "__main__":
    main()
