import pandas as pd
import os

def load_price_data(folder_path):
    data = {}

    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            name = file.replace(".csv", "")
            df = pd.read_csv(os.path.join(folder_path, file))

            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')

            data[name] = df.set_index('Date')['Price']

    combined = pd.DataFrame(data)
    return combined
