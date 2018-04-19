import pandas
import numpy


def sampling(df, num):
    rows = numpy.random.choice(df.index.values, num)
    return df.ix[rows]


def get_selected_cols(df, cols):
    return df[cols].copy()


def remove_nan(df, cols):
    for col in cols:
        df = df[pandas.notnull(df[col])]
    return df


def map_delay(df):
    delay_threshold = 15
    df['ARRIVAL_DELAY'] = pandas.cut(df['ARRIVAL_DELAY'], [-1000000, delay_threshold, 100000000], labels=['0', '1'])
    return df


def remove_num_airport(df):
    df = df[df['ORIGIN_AIRPORT'].apply(lambda x: not x.isdigit())]
    return df


def map_time(df):
    df['SCHEDULED_DEPARTURE'] = pandas.cut(df['SCHEDULED_DEPARTURE'],
                                           [0, 59, 159, 259, 359, 459, 559, 659, 759, 859, 959, 1059, 1159, 1259, 1359,
                                            1459, 1559, 1659, 1759, 1859, 1959, 2059, 2159, 2259, 2359],
                                           labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                                   20, 21, 22, 23])
    df['SCHEDULED_ARRIVAL'] = pandas.cut(df['SCHEDULED_ARRIVAL'],
                                         [0, 59, 159, 259, 359, 459, 559, 659, 759, 859, 959, 1059, 1159, 1259, 1359,
                                          1459, 1559, 1659, 1759, 1859, 1959, 2059, 2159, 2259, 2359],
                                         labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
                                                 20, 21, 22, 23])
    return df


def one_hot_encoding(df):
    categorical_cols = ['MONTH', 'DAY', 'DAY_OF_WEEK', 'AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
                        'SCHEDULED_DEPARTURE', 'SCHEDULED_ARRIVAL']
    df_new = pandas.get_dummies(df[categorical_cols], drop_first=False)
    df_new['DISTANCE'] = df['DISTANCE']
    return df_new


def main():
    flights_file = 'flights.csv'
    csv_path_flights = '../data/{}'.format(flights_file)
    x_cols = ['MONTH', 'DAY', 'DAY_OF_WEEK', 'AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
              'SCHEDULED_DEPARTURE', 'SCHEDULED_ARRIVAL', 'DISTANCE']
    y_col = ['ARRIVAL_DELAY']

    df_flights_raw = pandas.read_csv(csv_path_flights, dtype={'ORIGIN_AIRPORT': object, 'DESTINATION_AIRPORT': object})
    df_flights_raw = sampling(df_flights_raw, 10000)
    df_flights_raw = remove_nan(df_flights_raw, x_cols + y_col)
    df_flights_raw = remove_num_airport(df_flights_raw)

    df_data = get_selected_cols(df_flights_raw, x_cols)
    df_data = map_time(df_data)
    df_data = one_hot_encoding(df_data)

    df_delay = get_selected_cols(df_flights_raw, y_col)
    df_delay = map_delay(df_delay)

    df_data['ARRIVAL_DELAY'] = df_delay['ARRIVAL_DELAY']
    output_file_name = flights_file.split('.')[0] + '_one_hot.csv'
    df_data.to_csv('../data/{}'.format(output_file_name), encoding='utf-8')

if __name__ == '__main__':
    main()
