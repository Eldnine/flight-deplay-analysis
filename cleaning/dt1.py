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


def map_week_day(df):
    df['WEEKDAY'] = pandas.cut(df['DAY_OF_WEEK'], [0, 5, 7], labels=[1, 0])
    return df


def map_season(df):
    df['SEASON'] = pandas.cut(df['MONTH'], [0, 2, 5, 8, 11, 12], labels=[4, 1, 2, 3, 5])
    df.loc[df['SEASON'] == 5, ['SEASON']] = 4
    return df


def main():
    flights_file = 'flights.csv'
    csv_path_flights = '../data/{}'.format(flights_file)
    x_cols = ['YEAR', 'MONTH', 'DAY', 'DAY_OF_WEEK', 'AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
              'SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME']
    y_col = ['ARRIVAL_DELAY']

    df_flights_raw = pandas.read_csv(csv_path_flights, dtype={'ORIGIN_AIRPORT': object, 'DESTINATION_AIRPORT': object})
    df_flights_raw = sampling(df_flights_raw, 10000)
    df_flights_raw = remove_nan(df_flights_raw, x_cols + y_col)
    df_flights_raw = remove_num_airport(df_flights_raw)

    df_data = get_selected_cols(df_flights_raw, x_cols)
    df_data = map_time(df_data)
    df_data = map_week_day(df_data)
    df_data = map_season(df_data)

    df_delay = get_selected_cols(df_flights_raw, y_col)
    df_delay = map_delay(df_delay)

    df_data['ARRIVAL_DELAY'] = df_delay['ARRIVAL_DELAY']
    output_file_name = flights_file.split('.')[0] + '_dt.csv'
    df_data.to_csv('../data/{}'.format(output_file_name), encoding='utf-8')

if __name__ == '__main__':
    main()
