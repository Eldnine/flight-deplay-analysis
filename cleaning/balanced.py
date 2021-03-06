import pandas
import numpy


def sample_day(df):
    df = df[df['DAY'].apply(lambda x: x == 1 or x == 2)]
    return df


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
    categorical_cols = ['MONTH', 'DAY_OF_WEEK', 'AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
                        'SCHEDULED_DEPARTURE', 'SCHEDULED_ARRIVAL']
    df_new = pandas.get_dummies(df[categorical_cols])
    df_new['DISTANCE'] = df['DISTANCE']
    df_new['ARRIVAL_DELAY'] = df['ARRIVAL_DELAY']
    return df_new


def over_sampling(df):
    df_0 = df[df['ARRIVAL_DELAY'] == '0']
    df_1 = df[df['ARRIVAL_DELAY'] == '1']
    df_1 = df_1.append([df_1] * 4, ignore_index=True)
    df_upsampled = pandas.concat([df_0, df_1])
    print(df_upsampled['ARRIVAL_DELAY'].value_counts())

    return df_upsampled


def main():
    flights_file = 'flights.csv'
    csv_path_flights = '../data/{}'.format(flights_file)
    x_cols = ['MONTH', 'DAY', 'DAY_OF_WEEK', 'AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
              'SCHEDULED_DEPARTURE', 'SCHEDULED_ARRIVAL', 'DISTANCE']
    y_col = ['ARRIVAL_DELAY']

    df_flights_raw = pandas.read_csv(csv_path_flights, dtype={'ORIGIN_AIRPORT': object, 'DESTINATION_AIRPORT': object})
    df_flights_raw = sample_day(df_flights_raw)
    df_flights_raw = sampling(df_flights_raw, 12000)
    df_flights_raw = remove_nan(df_flights_raw, x_cols + y_col)
    df_flights_raw = remove_num_airport(df_flights_raw)

    df_data = get_selected_cols(df_flights_raw, x_cols + y_col)
    df_data = map_time(df_data)
    df_data = one_hot_encoding(df_data)

    df_data = map_delay(df_data)

    msk = numpy.random.rand(len(df_data)) < 0.8
    train = df_data[msk]
    test = df_data[~msk]

    train = over_sampling(train)

    output_file_name = flights_file.split('.')[0] + '_one_hot_balanced_train_3.csv'
    train.to_csv('../data/{}'.format(output_file_name), encoding='utf-8')
    output_file_name = flights_file.split('.')[0] + '_one_hot_balanced_test_3.csv'
    test.to_csv('../data/{}'.format(output_file_name), encoding='utf-8')

if __name__ == '__main__':
    main()
