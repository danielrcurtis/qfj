# This code is for part 1.3 of the Quantitative Finance Journey Series by Daniel R Curtis
# https://medium.com/@daniel.r.curtis/a-journey-in-quantitative-finance-df58cb88b159

import numpy as np
import tensorflow as tf
import platform
import pandas as pd
import os
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

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

def build_lstm_model(dataset, lstm_units, lstm_activation='relu', lstm_kernel_initializer='glorot_uniform', loss='mse', optimizer='nadam', metrics=['mae'], return_sequences=False):
    """
    Constructs an LSTM (Long Short-Term Memory) model using TensorFlow Keras. The model is composed of a single LSTM layer 
    followed by a Dense output layer with linear activation. It is tailored for sequence prediction tasks, 
    and the configuration of the LSTM layer and the compilation parameters can be customized.

    Args:
        dataset (tf.data.Dataset): The TensorFlow dataset that provides batches of input and target sequences.
                                   The input to the LSTM layer is expected to be in the form of a 3D tensor 
                                   with the shape (batch_size, time_steps, features).
        lstm_units (int): The number of neurons in the LSTM layer. This defines the dimensionality of the 
                          output space (i.e., the number of hidden states for each time step).
        lstm_activation (str, optional): Activation function to use in the LSTM layer. Defaults to 'relu', 
                                         which stands for rectified linear unit. Other common choices are
                                            'tanh' (hyperbolic tangent) and 'sigmoid'.
        lstm_kernel_initializer (str, optional): Initializer for the kernel weights matrix in the LSTM layer. 
                                                 Defaults to 'glorot_uniform', also known as Xavier uniform initializer.
        loss (str, optional): Loss function to be used during training. Defaults to 'mse' for mean squared error,
                              which is commonly used for regression tasks.
        optimizer (str, optional): Optimizer to use for training the model. Defaults to 'nadam', which is an 
                                   Adam optimization algorithm with Nesterov momentum.
        metrics (list, optional): List of metrics to be evaluated by the model during training and testing. 
                                  Defaults to ['mae'] for mean absolute error, which is a common metric for 
                                  regression tasks.

    Returns:
        tf.keras.Model: A compiled Keras model with an LSTM architecture, ready for training. The model has 
                        been compiled with the specified loss function, optimizer, and evaluation metrics.
    """
    # Determine the input shape for the LSTM layer from the first batch of the dataset
    for inputs, _ in dataset.take(1):
        input_shape = inputs.shape[1:]  # Input shape excluding the batch dimension

    # Define the model layers dynamically based on return_sequences
    layers = [
        tf.keras.layers.LSTM(
            lstm_units,
            activation=lstm_activation,
            kernel_initializer=lstm_kernel_initializer,
            input_shape=input_shape,
            return_sequences=return_sequences  # Controlled by the return_sequences argument
        )
    ]
    
    if return_sequences:
        layers.append(tf.keras.layers.Flatten())  # Add Flatten layer if return_sequences is True

    layers.append(tf.keras.layers.Dense(
        dataset.element_spec[1].shape[1],  # Number of neurons in the Dense layer matches the output dimension
        activation='linear'  # Linear activation function in the output layer for regression tasks
    ))

    # Define the LSTM model architecture
    model = tf.keras.models.Sequential(layers)

    # Compile the model with the specified optimizer, loss function, and evaluation metrics
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # Assuming a logger is initialized elsewhere, or initialize it here
    logger = logging.getLogger(__name__)
    logger.info(f"Built LSTM model with {lstm_units} LSTM units, {lstm_activation} activation, {lstm_kernel_initializer} kernel initializer, {loss} loss, {optimizer} optimizer, and {metrics} metrics")

    return model

def scale_dataset(data, scaler_type='standard'):
    """
    Scales the dataset based on the specified scaler type and returns the scaler object along with the scaled data.

    Args:
        data (np.array): The input data to scale.
        scaler_type (str): The type of scaler to use. Should be one of 'robust', 'standard', 'minmax'.

    Returns:
        tuple: A tuple containing the scaled dataset and the scaler object used for scaling.
    """
    if scaler_type == 'robust':
        scaler = RobustScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    elif scaler_type == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError(f"Scaler type '{scaler_type}' not recognized. Choose 'robust', 'minmax', or 'standard'.")

    # Fit the scaler on the data and then transform the data
    scaled_data = scaler.fit_transform(data)

    return scaled_data, scaler

def main():
    # Set the scaler type to use: 'robust', 'minmax', or 'standard'
    scaler_type = 'minmax' # Set this to scaler_type = None to disable scaling

    # Select the columns to be used for training our model
    selected_columns = ['close']

    # Read the data from the CSV file - the filename is hardcoded here based
    # on the example Binance information above. We also assume that you have
    # unzipped the downloaded file into the same path as it was downloaded.
    btc_data = pd.read_csv(os.path.join(download_dir, 'BTCUSDT-1m-2023-10.csv'))

    # Print information about the data
    print_dataframe_info(btc_data)

    # Select the columns to be used for training our model
    selected_data = select_df_columns(df=btc_data, columns=selected_columns)

    # Scale the data using the specified scaler type
    if scaler_type is not None:
        selected_data, scaler = scale_dataset(data=selected_data, scaler_type=scaler_type)

        # Create a DataFrame from the scaled data since the scaler returns a numpy array
        selected_data = pd.DataFrame(selected_data, columns=selected_columns)

        # Log details about the scaled data
        logger.info(f"Scaled data using {scaler_type} scaler.")
        if scaler_type == 'standard':
            logger.info(f"Scaler mean: {scaler.mean_}, Scaler variance: {scaler.var_}")
        elif scaler_type == 'minmax':
            logger.info(f"Scaler data min: {scaler.data_min_}, Scaler data max: {scaler.data_max_}")
        elif scaler_type == 'robust':
            logger.info(f"Scaler center: {scaler.center_}, Scaler scale: {scaler.scale_}")

    # Create a time series dataset. Consider changing the input_sequence_length and prediction_timesteps to see how the model performs.
    dataset = create_time_series_dataset(df=selected_data, input_sequence_length=11, prediction_timesteps=1,
                                          prediction_columns=['close'], check_dataset_stats=True, shuffle=True, stride=1)
    
    # Print the dataset shapes and scaler information.
    print_dataset_shapes(dataset)

    logger.info(f"Scaling method: {scaler_type}")

    # Split the dataset into training and validation sets.
    train_dataset, val_dataset = split_dataset(dataset=dataset, train_size_ratio=0.8)

    # Build the LSTM model and print the summary. Note that we have changed the number of LSTM units from 3 to 6 and the activation function from 'relu' to 'tanh'.
    # We have also change the loss function from 'mse' to 'mae' and enabled the return_sequences argument.
    # Have fun experimenting with different values!
    # lstm_activation options: 'relu', 'tanh', 'sigmoid'
    # loss options: 'mse', 'mae'
    # optimizer options: 'nadam', 'adam', 'sgd'
    # metrics options: 'mae', 'mse'
    # return_sequences options: True, False
    # lstm_kernel_initializer options: 'glorot_uniform', 'glorot_normal', 'he_uniform', 'he_normal'
    lstm_model = build_lstm_model(dataset=train_dataset, lstm_units=22, lstm_activation='tanh', lstm_kernel_initializer='glorot_uniform',
                                   loss='mae', optimizer='nadam', metrics=['mse'], return_sequences=False)
    lstm_model.summary()

    # Train the model
    history = lstm_model.fit(dataset, validation_data=val_dataset, epochs=5)

if __name__ == "__main__":
    main()
