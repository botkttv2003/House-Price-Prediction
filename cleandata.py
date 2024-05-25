import pandas as pd

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    # View the first 5 rows of the data set
    print(data.head())

    # View information of dataset
    print(data.info())

    # Check values that are null
    print(data.isna().sum())

    # Convert 'date' and 'time' to datetime
    data['date'] = pd.to_datetime(data['date'])
    data['time'] = pd.to_datetime(data['time'], format='%H:%M:%S').dt.time

    # Remove duplicate rows
    data = data.drop_duplicates()

    return data

def clean_data(data):
    # Check if there are hidden missing values
    hidden_missing_values = data.isin(['N/A', 'NA', 'NaN', 'None', 'Missing', '']).sum()

    # Check for negative values in 'price' and 'rooms'
    negative_price_indices = data[data['price'] <= 0].index
    negative_rooms_indices = data[data['rooms'] < -1].index

    # Remove rows with negative values
    data.drop(negative_price_indices, inplace=True)
    data.drop(negative_rooms_indices, inplace=True)

    return data

def save_data(data, output_file):
    data.to_csv(output_file, index=False)

if __name__ == "__main__":
    input_file = r"G:\University\BI\Final-project\all_v2.csv\all_v2.csv"
    output_file = 'data_newv2.csv'

    # Load data
    data = load_data(input_file)

    # Preprocess data
    data = preprocess_data(data)

    # Clean data
    data = clean_data(data)

    # Save cleaned data to a new CSV file
    save_data(data, output_file)
