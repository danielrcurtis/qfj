# This code is for part 1.2 of the Quantitative Finance Journey Series by Daniel R Curtis
# https://medium.com/@daniel.r.curtis/a-journey-in-quantitative-finance-eba36762688d

import numpy as np
import tensorflow as tf
import platform
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO) # Set the logging level to INFO
logger = logging.getLogger(__name__) # Get the logger for this file

# Set the path to the downloaded data on your computer
if platform.system() == "Windows":
    # Set the path to the downloaded data on Windows
    import winreg
    sub_key = r'SOFTWARE\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders'
    downloads_guid = '{374DE290-123F-4565-9164-39C4925E467B}'
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, sub_key) as key:
        location = winreg.QueryValueEx(key, downloads_guid)[0]
    download_dir = location
else:
    # Set the path to the downloaded data on Mac OS or Linux
    download_dir = os.path.join(os.path.expanduser('~'), 'downloads')

def print_dataframe_info(df : pd.DataFrame):
    """Prints information about a dataframe""
    Args:
        df (pd.DataFrame): The dataframe to print information about
    """
    logger.info("Dataframe info:")
    logger.info(df.info())
    logger.info("Dataframe description:")
    logger.info(df.describe())
    logger.info("Dataframe head:")
    logger.info(df.head())

def select_df_columns(df: pd.DataFrame, columns=['close']) -> pd.DataFrame:
    """
    Selects specified columns from a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): A list of column names to be selected. Defaults to ['close'].

    Returns:
        pd.DataFrame: A new DataFrame containing only the specified columns.
    """
    # Verify that all requested columns are in the DataFrame
    if not set(columns).issubset(df.columns):
        missing_columns = set(columns) - set(df.columns)
        raise ValueError(f"Columns not found in DataFrame: {missing_columns}")

    return df[columns]

def create_time_series_dataset(df: pd.DataFrame, input_sequence_length: int, prediction_timesteps: int,
                               prediction_columns: list, check_dataset_stats: bool = True, shuffle: bool = True,
                                 stride: int = None, batch_size: int = 8) -> tf.data.Dataset:
    """
    Prepares a pandas DataFrame for time series forecasting.

    Args:
        df (pd.DataFrame): The input dataframe.
        input_sequence_length (int): The length of the input sequence for the model.
        prediction_timesteps (int): The number of timesteps to predict.
        prediction_columns (list): List of columns to generate predictions for.
        check_dataset_stats (bool): If True, log the statistical information of the dataset.
        shuffle (bool): If True, shuffle the dataset.
        stride (int): The number of steps to move forward in the dataset after each sequence. Defaults to prediction_timesteps.
        batch_size (int): The batch size for the dataset. Defaults to 8.

    Returns:
        tf.data.Dataset: A TensorFlow dataset ready for time series forecasting.
    """

    if stride is None:
        stride = prediction_timesteps

    if not all(col in df.columns for col in prediction_columns):
        raise ValueError("Some prediction columns are not in the DataFrame")
    
    logger.info(f"Creating time series dataset with input sequence length {input_sequence_length}, prediction timesteps {prediction_timesteps}, prediction columns {prediction_columns}, stride {stride}")
    
    # Convert dataframe to numpy array for easier manipulation
    data = df.to_numpy()
    prediction_data = df[prediction_columns].to_numpy()

    # Prepare data for time series forecasting
    X, y = [], []
    for i in range(0, len(data) - input_sequence_length - prediction_timesteps + 1, stride):
        # The loop iterates over the DataFrame to create input-output sequence pairs for the dataset,
        # moving forward by 'stride' steps after each iteration.

        # 'i' is the starting index for each sequence in the dataset.
        # By using 'stride' in the range step, the starting index jumps forward by 'stride' positions
        # after processing each sequence, allowing for control over sequence overlap.

        # Append a sequence to X:
        # Extract a sequence of length 'input_sequence_length' from the DataFrame, starting at index 'i'.
        # This sequence acts as the input data for the model, representing a series of consecutive data points.
        X.append(data[i:(i + input_sequence_length)])

        # Append a corresponding sequence to y:
        # Extract a sequence for prediction, based on 'prediction_timesteps', immediately following the input sequence.
        # This sequence starts from index 'i + input_sequence_length' and extends 'prediction_timesteps' into the future.
        # These points are the target outputs for the model, representing the values it needs to predict.
        y.append(prediction_data[i + input_sequence_length:i + input_sequence_length + prediction_timesteps])

    X, y = np.array(X), np.array(y)

    if check_dataset_stats:
        # Calculate statistics for input sequences
        mean_X, std_X = np.mean(X), np.std(X)
        min_X, max_X = np.min(X), np.max(X)

        # Calculate statistics for output sequences
        mean_y, std_y = np.mean(y), np.std(y)
        min_y, max_y = np.min(y), np.max(y)

        # Log the statistics
        logger.info(f"Input Sequence Statistics - Mean: {mean_X}, Standard Deviation: {std_X}, Min: {min_X}, Max: {max_X}")
        logger.info(f"Output Sequence Statistics - Mean: {mean_y}, Standard Deviation: {std_y}, Min: {min_y}, Max: {max_y}")


    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_tensor_slices((X, y))

    if shuffle:
        # Shuffle the dataset
        dataset = dataset.shuffle(buffer_size=len(X))
        logger.info(f"Dataset shuffled using a buffer size of {len(X)}")

    # Batch the dataset
    dataset = dataset.batch(batch_size)
    logger.info(f"Dataset batched with batch size of {batch_size}")

    return dataset

def print_dataset_shapes(dataset: tf.data.Dataset):
    """
    Prints the shapes of the input and output sequences in the TensorFlow dataset.

    Args:
        dataset (tf.data.Dataset): The TensorFlow dataset to print shapes from.
    """
    for input_seq, output_seq in dataset.take(1):
        logger.info(f"Input Sequence Shape: {input_seq.shape}")
        logger.info(f"Output Sequence Shape: {output_seq.shape}")

def split_dataset(dataset, train_size_ratio=0.8):
    """
    Splits a batched TensorFlow dataset into training and validation datasets.

    Args:
        dataset (tf.data.Dataset): The batched TensorFlow dataset to split.
        train_size_ratio (float): The proportion of the dataset to use for training (between 0 and 1).

    Returns:
        tf.data.Dataset: The training dataset.
        tf.data.Dataset: The validation dataset.
    """
    # Determine the number of batches in the dataset
    total_batches = len(list(dataset))

    # Calculate the number of batches for the training dataset
    train_batches = int(total_batches * train_size_ratio)

    # Split the dataset
    train_dataset = dataset.take(train_batches)
    val_dataset = dataset.skip(train_batches)

    return train_dataset, val_dataset

def build_lstm_model(dataset, lstm_units):
    """
    Builds a simple LSTM model using TensorFlow Keras based on the LSTM unit size and dataset dimensions,
    with linear activation functions.

    Args:
        dataset (tf.data.Dataset): The TensorFlow dataset to get input and output dimensions.
        lstm_units (int): The number of units in the LSTM layer.

    Returns:
        tf.keras.Model: A compiled Keras LSTM model.
    """
    # Determine the input shape from the dataset
    for inputs, _ in dataset.take(1):
        # The shape of inputs is expected to be (batch_size, time_steps, features)
        # Ensure the input shape for LSTM layer is (time_steps, features)
        input_shape = inputs.shape[1:]

    # Model definition
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(lstm_units, activation='linear', input_shape=input_shape, return_sequences=False),
        tf.keras.layers.Dense(dataset.element_spec[1].shape[1], activation='linear')
    ])

    # Compile the model
    model.compile(optimizer='nadam', loss='mse', metrics=['mae'])

    return model

def main():
    # Read the data from the CSV file - the filename is hardcoded here based
    # on the example Binance information above. We also assume that you have
    # unzipped the downloaded file into the same path it was downloaded into.
    btc_data = pd.read_csv(os.path.join(download_dir, 'BTCUSDT-1m-2023-10.csv'))

    # Print information about the data
    print_dataframe_info(btc_data)

    # Select the columns to be used for training our model
    selected_data = select_df_columns(df=btc_data, columns=['close'])

    # Create a time series dataset
    dataset = create_time_series_dataset(df=selected_data, input_sequence_length=3, prediction_timesteps=1,
                                          prediction_columns=['close'], check_dataset_stats=True, shuffle=True, stride=1)
    # Print the dataset shapes
    print_dataset_shapes(dataset)

    # Split the dataset into training and validation sets
    train_dataset, val_dataset = split_dataset(dataset=dataset, train_size_ratio=0.8)

    # Build the LSTM model and print the summary
    lstm_model = build_lstm_model(dataset=train_dataset, lstm_units=3)
    lstm_model.summary()

    # Train the model
    history = lstm_model.fit(dataset, validation_data=val_dataset, epochs=5)

if __name__ == "__main__":
    main()
