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


def one_hot_encoding(df, non_categorical_cols):
    result = pandas.DataFrame(columns=non_categorical_cols)
    categorical_cols = []
    for col in filter(lambda p: p not in non_categorical_cols, df.columns):
        categorical_cols.append(col)
    dumdata = pandas.get_dummies(df[categorical_cols], drop_first=False)
    for col in non_categorical_cols:
        result[col] = df[col]
    for col in dumdata.columns:
        result[col] = dumdata[col]
    return result


def map_delay(df):
    delay_threshold = 30
    df['ARRIVAL_DELAY'] = pandas.cut(df['ARRIVAL_DELAY'], [-1000000, delay_threshold, 100000000], labels=['0', '1'])
    return df


def remove_num_airport(df):
    df = df[df['ORIGIN_AIRPORT'].apply(lambda x: not x.isdigit())]
    return df


def map_time(df):
    df['SCHEDULED_DEPARTURE'] = pandas.cut(df['SCHEDULED_DEPARTURE'],
                                     [0, 359, 759, 1159, 1559, 1959, 2359],
                                     labels=[1, 2, 3, 4, 5, 6])
    return df


def main():
    flights_file = 'flights.csv'
    csv_path_flights = '../data/{}'.format(flights_file)
    x_cols = ['YEAR', 'MONTH', 'DAY', 'DAY_OF_WEEK', 'AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
              'SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME']
    y_col = ['ARRIVAL_DELAY']
    non_categorical_cols = ['SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME']

    df_flights_raw = pandas.read_csv(csv_path_flights, dtype={'ORIGIN_AIRPORT': object, 'DESTINATION_AIRPORT': object})
    df_flights_raw = sampling(df_flights_raw, 10000)
    df_flights_raw = remove_nan(df_flights_raw, x_cols + y_col)
    df_flights_raw = remove_num_airport(df_flights_raw)

    df_x = get_selected_cols(df_flights_raw, x_cols)
    # df_x = one_hot_encoding(df_x, non_categorical_cols)
    df_x = map_time(df_x)
    # output_file_name = flights_file.split('.')[0] + '_dt_x.csv'
    # df_x.to_csv('../data/{}'.format(output_file_name), encoding='utf-8')


    df_delay = get_selected_cols(df_flights_raw, y_col)
    df_delay = map_delay(df_delay)
    # output_file_name = flights_file.split('.')[0] + '_dt_y.csv'
    # df_delay.to_csv('../data/{}'.format(output_file_name), encoding='utf-8')

    df_x['ARRIVAL_DELAY'] = df_delay['ARRIVAL_DELAY']
    output_file_name = flights_file.split('.')[0] + '_dt.csv'
    df_x.to_csv('../data/{}'.format(output_file_name), encoding='utf-8')

if __name__ == '__main__':
    main()
