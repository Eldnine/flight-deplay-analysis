import pandas
import numpy


def get_selected_cols(csv_path, cols, shadow_col):
    df_flights = pandas.read_csv(csv_path)
    df_flights = remove_nan(df_flights, cols, shadow_col)
    return df_flights[cols].copy()


def remove_nan(df, x_cols, y_col):
    for col in x_cols + y_col:
        df = df[pandas.notnull(df[col])]
    return df


def one_hot_encoding(df, non_categorical_cols):
    result = pandas.DataFrame(columns=non_categorical_cols)
    categorical_cols = []
    for col in filter(lambda p: p not in non_categorical_cols, df.columns):
        categorical_cols.append(col)
    dumdata = pandas.get_dummies(df, columns=categorical_cols, drop_first=False)
    for col in non_categorical_cols:
        result[col] = df[col]
    for col in dumdata.columns:
        result[col] = dumdata[col]
    return result


def map_delay(df):
    delay_threshold = 30
    df['ARRIVAL_DELAY'] = pandas.cut(df['ARRIVAL_DELAY'], [-1000000, delay_threshold, 100000000], labels=['0', '1'])
    return df


def main():
    flights_file = 'flights-samples.csv'
    csv_path_flights = '../data/{}'.format(flights_file)
    x_cols = ['YEAR', 'MONTH', 'DAY', 'DAY_OF_WEEK', 'AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
              'SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME']
    y_col = ['ARRIVAL_DELAY']
    non_categorical_cols = ['SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME']

    df_flights = get_selected_cols(csv_path_flights, x_cols, y_col)
    df_flights = one_hot_encoding(df_flights, non_categorical_cols)
    output_file_name = flights_file.split('.')[0] + '_one_hot.csv'
    df_flights.to_csv('../data/{}_x'.format(output_file_name), encoding='utf-8')

    df_delay = get_selected_cols(csv_path_flights, y_col, [])
    df_delay = map_delay(df_delay)
    df_delay.to_csv('../data/{}_y'.format(output_file_name), encoding='utf-8')


if __name__ == '__main__':
    main()
