import pandas
import numpy as np
import matplotlib.pyplot as plt


def main():
    csv_flights = './flights-samples.csv'
    csv_airlines = './airlines.csv'
    csv_airports = './airports.csv'

    df_flights = pandas.read_csv(csv_flights)
    df_airlines = pandas.read_csv(csv_airlines)
    df_airports = pandas.read_csv(csv_airports)
    print('1. Number of records and attributes', df_flights.shape)
    print('2. Get data types of attributes\n', df_flights.dtypes)
    #print('2. Basic info of flights \n', df_flights.describe())
    df_flights['DATE'] = pandas.to_datetime(df_flights[['YEAR', 'MONTH', 'DAY']])


if __name__ == '__main__':
    main()
