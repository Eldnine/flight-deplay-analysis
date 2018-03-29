import pandas
import numpy
import matplotlib.pyplot as plt
import datetime


def float_to_time(time_float):
    if pandas.isnull(time_float):
        return numpy.nan
    else:
        if time_float == 2400:
            time_float = 0
        time_str = "{0:04d}".format(int(time_float))
        time = datetime.time(int(time_str[0:2]), int(time_str[2:4]))
        return time


def df_col_float_to_time(df, col_name):
    results = []
    for index, cols in df[[col_name]].iterrows():
        results.append(float_to_time(cols[0]))
    return pandas.Series(results)


def main():
    csv_flights = '../data/flights-samples.csv'
    csv_airlines = '../data/airlines.csv'
    csv_airports = '../data/airports.csv'

    df_flights = pandas.read_csv(csv_flights)
    df_airlines = pandas.read_csv(csv_airlines)
    df_airports = pandas.read_csv(csv_airports)
    print('1. Number of records and attributes', df_flights.shape)
    print('2. Get data types of attributes\n', df_flights.dtypes)

    df_flights['DATE'] = pandas.to_datetime(df_flights[['YEAR', 'MONTH', 'DAY']])

    df_flights['SCHEDULED_DEPARTURE'] = df_col_float_to_time(df_flights, 'SCHEDULED_DEPARTURE')
    df_flights['DEPARTURE_TIME'] = df_col_float_to_time(df_flights, 'DEPARTURE_TIME')
    df_flights['SCHEDULED_ARRIVAL'] = df_col_float_to_time(df_flights, 'SCHEDULED_ARRIVAL')
    df_flights['ARRIVAL_TIME'] = df_col_float_to_time(df_flights, 'ARRIVAL_TIME')

    missing_df = df_flights.isnull().sum(axis=0).reset_index()
    missing_df.columns = ['Column Name', 'Number of Missing']
    missing_df['Filling Factor (%)'] = (df_flights.shape[0] - missing_df['Number of Missing'])\
                                       / df_flights.shape[0] * 100
    missing_df.sort_values('Filling Factor (%)').reset_index(drop=True)
    print(missing_df)

if __name__ == '__main__':
    main()
