import pandas
import numpy


def get_selected_cols(csv_path, cols):
    df_flights = pandas.read_csv(csv_path)
    return df_flights[cols].copy()


def remove_nan(df):
    for col in df.columns:
        df = df[pandas.notnull(df[col])]
    return df


def convert_one_hot(df, non_categorical_cols):
    # one_hot_matrices = []
    categorical_cols = []
    for col in filter(lambda p: p not in non_categorical_cols, df.columns):
        categorical_cols.append(col)
    dumdata = pandas.get_dummies(df, columns=categorical_cols, drop_first=False)
    for col in filter(lambda p: p in non_categorical_cols, df.columns):
        dumdata[col] = dumdata[col].values.reshape(-1, 1)
    dumdata.reindex(sorted(dumdata.columns))
    return dumdata


def main():
    csv_path_flights = '../data/flights-samples.csv'
    cols = ['YEAR', 'MONTH', 'DAY', 'DAY_OF_WEEK', 'AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
            'SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME', 'ARRIVAL_DELAY']
    non_categorical_cols = ['SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME',
                            'ARRIVAL_DELAY']

    df_flights = get_selected_cols(csv_path_flights, cols)
    df_flights = remove_nan(df_flights)
    df_flights = convert_one_hot(df_flights, non_categorical_cols)
    df_flights.to_csv('../data/flight_one_hot.csv', sep='\t', encoding='utf-8')


if __name__ == '__main__':
    main()
