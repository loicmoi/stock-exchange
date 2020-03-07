#! /usr/bin/env python3
# coding: utf-8

"""
Summary
-------
Analyze the statistics of a stock or a fund and estimate from
its historical behavior the fund forecast potential.

Inputs
------
* Fund choice between 1 and 5.
* Fund period range between two dates.
* Fund statistics parameters such as fund name and acronym.

Outputs
-------
* Fund statistics and forecast potential.
* Fund regression line coefficients.

Example
-------
launch_analysis(1, ('2010-01-01', '2020-01-01'))
"""

############# Standard modules import:

from os import path
import logging as lg

############# Logging module initialization:

from logging.handlers import TimedRotatingFileHandler

MY_FORMAT = "%(asctime)-24s %(levelname)-6s %(message)s"
lg.basicConfig(format=MY_FORMAT, level=lg.INFO)
my_logger = lg.getLogger()

my_logger.info("General modules and logger format are initialized.")

############# Specific modules import:

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

############# Internal modules import:

############# Classes definition:

class DataSet:
    """Class that creates a Pandas dataframe object with data science methods."""
    def __init__(self, name):
        self.name = name
    
    def __str__(self):
        """Special method called when converting the dataset object to a string or
        calling the 'print' function.
        """
        return "{}:\n {}".format(self.name, self.dataframe)
    
    def __repr__(self):
        """Special method allowing the display when entering the dataset object in
        the interpreter.
        """
        return "{}".format(self.dataframe)
    
    def load_params_from_csv(self, csv_choice, csv_path, csv_sep=','):
        """Load parameters from CSV path."""
        my_logger.info("Opening data path {}".format(csv_path))
        csv_df = pd.read_csv(csv_path, sep=csv_sep)
        if(csv_choice in range (1, 6)):
            self.name = csv_df.loc[csv_choice - 1, 'fund_name']
            self.acronym = csv_df.loc[csv_choice - 1, 'fund_acronym']
        else:
            my_logger.error("Fund "+str(csv_choice)+" choice is not correct, please choose between 1 and 5.")
            return None
    
    def load_data_from_csv(self, csv_path, csv_sep=','):
        """Load dataframe from CSV path."""
        my_logger.info("Opening data path {}".format(csv_path))
        self.dataframe = pd.read_csv(csv_path, sep=csv_sep)
    
    def load_data_from_dataframe(self, dataframe):
        """Load dataframe from another dataframe object."""
        self.dataframe = dataframe
    
    def load_data_from_web(self, acronym, start_date, end_date):
        """Load dataframe from the web."""
        import pandas_datareader as pdr
        self.dataframe = pdr.DataReader(acronym, data_source='yahoo', start=start_date, end=end_date)
        self.dataframe.reset_index(inplace=True)
        self.dataframe['Date'] = self.dataframe['Date'].astype(str)
    
    def clean_dataframe(self):
        """Clean the dataframe by removing rows with missing values such as 'NaN'
        and none relevant columns.
        """
        self.dataframe.dropna(inplace=True)
        self.dataframe.drop(columns='Open', inplace=True)
        self.dataframe.drop(columns='High', inplace=True)
        self.dataframe.drop(columns='Low', inplace=True)
        self.dataframe.drop(columns='Adj Close', inplace=True)
        self.dataframe.drop(columns='Volume', inplace=True)
        self.dataframe.rename(columns={'Close':'daily_value'}, inplace=True)
        self.dataframe.rename(columns={'Date':'date'}, inplace=True)
    
    def add_datetime_column(self):
        """Add a 'datetime' column created from the 'date' column and remove rows
        with missing values such as 'NaT'.
        """
        date_only_df = self.dataframe.drop(columns=['daily_value'])
        date_only_df['year'] = self.dataframe['date'].str.split('-').str[0]
        date_only_df['month'] = self.dataframe['date'].str.split('-').str[1]
        date_only_df['day'] = self.dataframe['date'].str.split('-').str[2]
        date_only_df.drop(columns=['date'], inplace=True)
        self.dataframe['datetime'] = pd.to_datetime(date_only_df, errors='coerce', yearfirst=True)
        self.dataframe.dropna(inplace=True)
    
    def add_date_of_week_column(self):
        """Add a 'day_of_week' (Monday=0 ... Sunday=6) column from  the 'datetime'
        column.
        """
        self.dataframe['day_of_week'] = self.dataframe['datetime'].dt.dayofweek
    
    def add_last_day_of_week_column(self):
        """Add a 'last_day_of_week' boolean column from  the 'datetime' column."""
        last_day_of_week_ser = self.dataframe['datetime'].groupby(self.dataframe.datetime.dt.to_period('W')).apply(lambda x: x.values.max())
        self.dataframe['last_day_of_week'] = self.dataframe['datetime'].isin(last_day_of_week_ser)
    
    def add_last_day_of_month_column(self):
        """Add a 'last_day_of_month' boolean column from  the 'datetime' column."""
        last_day_of_month_ser = self.dataframe['datetime'].groupby(self.dataframe.datetime.dt.to_period('M')).apply(lambda x: x.values.max())
        self.dataframe['last_day_of_month'] = self.dataframe['datetime'].isin(last_day_of_month_ser)
    
    def add_last_day_of_year_column(self):
        """Add a 'last_day_of_year' boolean column from  the 'datetime' column."""
        last_day_of_year_ser = self.dataframe['datetime'].groupby(self.dataframe.datetime.dt.to_period('Y')).apply(lambda x: x.values.max())
        self.dataframe['last_day_of_year'] = self.dataframe['datetime'].isin(last_day_of_year_ser)
    
    def add_daily_gain_columns(self):
        """Add a 'abs_daily_gain' and 'rel_daily_gain' columns from the 'daily_value'
        column difference between the previous day."""
        self.dataframe['abs_daily_gain'] = self.dataframe['daily_value'].diff(periods=1)
        self.dataframe['rel_daily_gain'] = self.dataframe['abs_daily_gain'] / self.dataframe['daily_value'].shift(periods=1)
    
    def add_weekly_value_and_gain_columns(self):
        """Add a 'abs_weekly_gain' and 'rel_weekly_gain' columns from the 'daily_value'
        column difference between the previous week."""
        df_week_ser = self.dataframe['daily_value'].where(self.dataframe.last_day_of_week == True)
        df_week_ser.dropna(inplace=True)
        self.dataframe['weekly_value'] = df_week_ser
        self.dataframe['abs_weekly_gain'] = df_week_ser.diff(periods=1)
        self.dataframe['rel_weekly_gain'] = self.dataframe['abs_weekly_gain'] / df_week_ser.shift(periods=1)
        self.dataframe.drop(columns=['last_day_of_week'], inplace=True)
    
    def add_monthly_value_and_gain_columns(self):
        """Add a 'abs_monthly_gain' and 'rel_monthly_gain' columns from the 'daily_value'
        column difference between the previous month."""
        df_month_ser = self.dataframe['daily_value'].where(self.dataframe.last_day_of_month == True)
        df_month_ser.dropna(inplace=True)
        self.dataframe['monthly_value'] = df_month_ser
        self.dataframe['abs_monthly_gain'] = df_month_ser.diff(periods=1)
        self.dataframe['rel_monthly_gain'] = self.dataframe['abs_monthly_gain'] / df_month_ser.shift(periods=1)
        self.dataframe.drop(columns=['last_day_of_month'], inplace=True)
    
    def add_yearly_value_and_gain_columns(self):
        """Add a 'abs_yearly_gain' and 'rel_yearly_gain' columns from the 'daily_value'
        column difference between the previous year."""
        df_year_ser = self.dataframe['daily_value'].where(self.dataframe.last_day_of_year == True)
        df_year_ser.dropna(inplace=True)
        self.dataframe['yearly_value'] = df_year_ser
        self.dataframe['abs_yearly_gain'] = df_year_ser.diff(periods=1)
        self.dataframe['rel_yearly_gain'] = self.dataframe['abs_yearly_gain'] / df_year_ser.shift(periods=1)
        self.dataframe.drop(columns=['last_day_of_year'], inplace=True)
    
    def rearrange_dataframe(self):
        """Rearrange the dataframe by moving columns in a right order."""
        self.dataframe = self.dataframe.reindex(columns=['date', 'datetime',
                                                         'daily_value', 'abs_daily_gain', 'rel_daily_gain',
                                                         'weekly_value', 'abs_weekly_gain', 'rel_weekly_gain',
                                                         'monthly_value', 'abs_monthly_gain', 'rel_monthly_gain',
                                                         'yearly_value', 'abs_yearly_gain', 'rel_yearly_gain'])
    
    def get_mean(self, column_name):
        """Return the mean of the values for the requested axis."""
        data_mean = self.dataframe[column_name].mean(axis=0)
        return data_mean
    
    def get_variance(self, column_name):
        """Return unbiased variance over requested axis."""
        data_var = self.dataframe[column_name].var(axis=0)
        return data_var
    
    def get_standard_deviation(self, column_name):
        """Return sample standard deviation over requested axis."""
        data_std = self.dataframe[column_name].std(axis=0)
        return data_std
    
    def get_median(self, column_name):
        """Return the median of the values for the requested axis."""
        data_median = self.dataframe[column_name].median(axis=0)
        return data_median
    
    def display_histogram(self, x_column, period_range=('', 'all'), plotsize=(17, 4), data_color='black'):
        """Display a histogram of x data."""
        # Get x variable in a specific period range from the dataset
        period_range_df = self.dataframe.set_index('date').loc[period_range[0]:period_range[1]]
        x = period_range_df[x_column].dropna()
        if('rel' in x_column):
            x = x * 100
            data_ratio = " (%)"
        else:
            data_ratio = ""
        
        # Visualize the dataset histogram
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=plotsize)
        
        ax.hist(x, bins=100, normed=False, label=x_column, color=data_color)
        
        ax.set_title(str(x_column.capitalize())+" histogram")
        ax.set_xlabel(x_column.capitalize()+data_ratio)
        ax.set_ylabel("Occurrence quantity")
        
        min_ylim, max_ylim = plt.ylim()
        ax.axvline(x.mean(), color='black', linestyle='dashed', linewidth=1)
        ax.text(x.mean()*1.05+1, max_ylim*0.9, "Mean: {:.3f}".format(x.mean())+data_ratio)
    
    def display_chart(self, x_column, y_column, period_range=('', 'all'), function='plot',
                      data_scale='linear', plotsize=(17, 4), data_color='black'):
        """Display y vs x data as a chart with varying functions, marker size
        and/or color:
        - 'plot' display the graph as a continuous line.
        - 'scatter' display the graph as a collection of points.
        - 'bar' display the graph as a series of bars.
        """
        from pandas.plotting import register_matplotlib_converters
        register_matplotlib_converters()
        
        # Get the x and y variables in a specific period range from the dataset
        period_range_df = self.dataframe.set_index('date').loc[period_range[0]:period_range[1]]
        x = period_range_df[x_column][period_range_df[y_column].notna()]
        y = period_range_df[y_column].dropna()
        if(data_scale == 'log10'):
            y = np.log10(y)
        
        # Visualize the dataset chart
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=plotsize)
        
        if(function == 'scatter'):
            ax.scatter(x, y, color=data_color)
        elif(function == 'bar'):
            ax.bar(x, y, color=data_color)
        else:
            ax.plot(x, y, color=data_color)
        
        ax.set_title(y_column.capitalize()+" vs "+x_column.capitalize())
        ax.set_xlabel(x_column.capitalize())
        ax.set_ylabel(y_column.capitalize()+"\n(Data scale: "+data_scale+")")
    
    def display_polynomial_regression(self, x_column, y_column, period_range=('', 'all'),
                                      function='scatter', data_scale='linear', plotsize=(17, 4),
                                      data_color='black', regression_line_color='gray',
                                      polynomial_degree=3, show_stats=False):
        """Display the Polynomial Regression of data.
        
        LinearRegression fits a polynomial model with coefficients w = (w1, …, wp) to
        minimize the residual sum of squares between the observed targets in the
        dataset, and the targets predicted by the linear approximation.
        """
        from pandas.plotting import register_matplotlib_converters
        register_matplotlib_converters()
        
        # Get the X and y variables in a specific period range from the dataset
        period_range_df = self.dataframe.set_index('date').loc[period_range[0]:period_range[1]]
        X = period_range_df[x_column][period_range_df[y_column].notna()].values.reshape(-1, 1)
        y = period_range_df[y_column].dropna().values
        if(data_scale == 'log10'):
            y = np.log10(y)
        
        # Fit Polynomial Regression to the dataset
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        poly = PolynomialFeatures(degree=polynomial_degree)
        X_poly = poly.fit_transform(X)
        
        poly.fit(X_poly, y)
        poly_regr = LinearRegression()
        poly_regr.fit(X_poly, y)
        y_polynomial_predictor = poly_regr.predict(poly.fit_transform(X))
        
        # Visualize the Polynomial Regression results
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=plotsize)
        
        if(function == 'plot'):
            ax.plot(X, y, color=data_color, label='Original data')
        else:
            ax.scatter(X, y, color=data_color, label='Original data')
        
        ax.plot(X, y_polynomial_predictor, color=regression_line_color, label='Fitted line')
        ax.set_title("Polynomial regression")
        ax.set_xlabel(x_column.capitalize())
        ax.set_ylabel(y_column.capitalize()+"\n(Data scale: "+data_scale+")")
        
        if show_stats:
            # The coefficients: y = β0 + β1*x + β2*x^2 + … + βd*x^d + Ɛ
            print("Coefficients: y = {} + {}*x^d.".format(poly_regr.intercept_, poly_regr.coef_))
            # The mean absolute error: MAE
            print("Mean Absolute Error: {}".format(mean_absolute_error(y, y_polynomial_predictor)))
            # The mean squared error: MSE
            print("Mean Squared Error: {}".format(mean_squared_error(y, y_polynomial_predictor)))
            # The root mean squared error: RMSE
            print("Root Mean Squared Error: {}".format(np.sqrt(mean_squared_error(y, y_polynomial_predictor))))
            # The coefficient of determination: R^2. 1 is perfect prediction
            print("R^2 coefficient of determination: {}".format(r2_score(y, y_polynomial_predictor)))

my_logger.info("General classes are defined.")

############# Functions definition:

############# Analysis program definition:

def launch_analysis(fund_choice, fund_period_range):
    """Launch analysis of fund weighted average gain and potential from fund choice and period range.
    
    Parameters
    ----------
    fund_choice : int
        Int value of the fund choice between 1 and 5.
    fund_period_range : tuple
        Tuple of the analysis period range between two dates.
    
    Returns
    -------
    fund_dataset : Dataset object
        Dataset object corresponding to the fund dataframe values and gains for different periods.
    
    Raises
    ------
    Exception
        Error exception raised when the analysis can't be launched.
    """
    my_logger.debug("The selected fund_choice is '"+str(fund_choice)+"'.")
    my_logger.debug("The selected fund_period_range is '"+str(fund_period_range)+"'.")
    try:
        my_logger.info("Initialize input data for the analysis.")
        
        ############# Input data initialization of fund_dataset object:
        
        fund_dataset = DataSet("fun_dataset")
        fund_dataset.load_params_from_csv(csv_path='data/fund_statistics_params.csv', csv_choice=fund_choice)
        fund_dataset.load_data_from_web(acronym=fund_dataset.acronym, start_date=fund_period_range[0], end_date=fund_period_range[1])
        my_logger.info("Input data of '"+str(fund_dataset.name)+"' fund is initialized.")
        
        my_logger.info("Let's start the analysis...")
        
        ############# Clean, add different periods gain columns and rearrange the fund_dataset dataframe:
        
        fund_dataset.clean_dataframe()
        
        fund_dataset.add_datetime_column()
        fund_dataset.add_last_day_of_week_column()
        fund_dataset.add_last_day_of_month_column()
        fund_dataset.add_last_day_of_year_column()
        
        fund_dataset.add_daily_gain_columns()
        fund_dataset.add_weekly_value_and_gain_columns()
        fund_dataset.add_monthly_value_and_gain_columns()
        fund_dataset.add_yearly_value_and_gain_columns()
        
        fund_dataset.rearrange_dataframe()
        
        my_logger.info("Absolute and relative gain columns for different periods has been added.")
        
        ############# Return the fund_dataset object:
        
        my_logger.info("Return the fund dataset object to display statistics and regression line.")
        
        return fund_dataset
    
    except Exception as e:
        my_logger.error("Can't launch the analysis.\n* ExceptionError: {}".format(e))

############# Args function:

def parse_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--choice", help="""Fund statistics analysis choice between 1 and 5""")
    parser.add_argument("-pr", "--period_range", help="""Fund statistics analysis period range between two dates""")
    parser.add_argument("-d", "--debug", action='store_true', help="""Start debugging the application""")
    return parser.parse_args()

############# Main function:

def main():
    args = parse_arguments()
    if args.debug:
        my_logger.basicConfig(format=MY_FORMAT, level=logging.DEBUG)
        my_logger.debug("Let's start to enter in debug mode")
        import pdb; pdb.set_trace() # pdb command line in case of debugging
    my_logger.info("Start fund statistics analysis.")
    launch_analysis(int(args.choice), args.period_range)
    my_logger.info("Stop fund statistics analysis.")

if __name__ == "__main__":
    main()
