from binance.client import Client
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import numpy as np
from matplotlib.ticker import MultipleLocator
from scipy import stats, signal
from functools import partial
import talib as ta
pio.renderers.default='browser'


#%%

class Data():
    
    def __init__(self):
        api_key = 'your_api_key'
        api_secret = 'your_api_secret'
        self.client = Client(api_key, api_secret)
        self.symbol = 'BTCUSDT'
        self.interval = '1h'  # Daily data
        self.start_str = '2024-06-01'  # Start date
        self.end_str = None # using None gives current time GMT
        self.metrics = []
        
        
    def get_historical_klines(self, show_dataframe=False):
        
        """
        Retrieves historical kline (candlestick) data from Binance for a specified trading symbol and interval,
        covering a specific date range. The data is transformed into a pandas DataFrame, which can be optionally
        displayed in the console.
    
        Attributes (class attributes, not parameters):
        - symbol (str): The symbol to fetch data for (e.g., 'BTCUSDT'). Must be set as a class attribute before calling this method.
        - interval (str): The interval for kline data (e.g., '1d' for daily data). Must be set as a class attribute.
        - start_str (str): The start date string in format 'YYYY-MM-DD'. Must be set as a class attribute.
        - end_str (str): The end date string in format 'YYYY-MM-DD'. Optional; must be set as a class attribute if used.
    
        Parameters:
        - show_dataframe (bool, optional): If set to True, prints the resulting DataFrame to the console after fetching and processing the data.
        
        Returns:
        - DataFrame: A pandas DataFrame containing the kline data formatted with relevant financial metrics.
    
        Example Usage:
        Assuming an instance `binance_data` with necessary attributes set, you might call:
        binance_data.get_historical_klines(show_dataframe=True)
    
        Processing Steps:
        - The method fetches the kline data using Binance's API, provided the `client` is authenticated and ready.
        - The raw data, initially in list form, is converted into a pandas DataFrame with specified column names.
        - Timestamps (originally in milliseconds) are converted to more readable datetime objects.
        - The DataFrame is cleaned by setting timestamps as the index and dropping columns that are not needed for further analysis.
        - Data types are converted to float for numerical processing.
        - If `show_dataframe` is True, the symbol and DataFrame are printed to provide a visual confirmation of the data fetched.
        """
        
        # Fetch kline data from Binance API
        klines = self.client.get_historical_klines(self.symbol, self.interval, self.start_str, self.end_str)
    
        # Create a DataFrame from the raw kline data
        self.df = pd.DataFrame(klines, columns=[
            'Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'close_time',
            'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume', 'ignore'
        ])
    
        # Convert timestamp column to datetime format for better readability
        self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'], unit='ms')
    
        # Set the timestamp column as the index of the DataFrame for easier time series analysis
        self.df.set_index('Timestamp', inplace=True)
    
        # Remove columns that are not necessary for the user's analysis to streamline the DataFrame
        self.df.drop(['close_time', 'quote_asset_volume', 'number_of_trades', 
                      'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 
                      'ignore'], axis=1, inplace=True)
    
        # Ensure all data in the DataFrame is of float type for consistency in numerical calculations
        self.df = self.df.astype(float)
    
        # Optionally display the DataFrame to the console for verification or inspection
        if show_dataframe:
            print(self.symbol)
            print(self.df)
        
    
    def add(self,colname,col):
        if colname not in self.metrics:
            self.df[colname] = col
            self.metrics.append(colname)
            print(self.df)
        else:
            print('Metric already available in dataframe')
        
        
    def remove(self,metric):
        self.df.drop(metric, axis = 1, inplace = True)
        self.df.drop(metric+'_Range_Signal', axis = 1, inplace = True)
        self.metrics.remove(metric)
        print(self.df)
        print(self.metrics)
        
        
    def create_bounds(self, func, name, step, window_size, dynamic=False, **func_kwargs):
        """
        Modifies the class's DataFrame by appending new columns based on the outputs from a specified function.
        This method supports both static (non-dynamic) and dynamic modes of operation.
        
        Parameters:
        - func (callable): A function expected to return a pandas DataFrame, which takes at least a `window_size` parameter
                           and can accept additional keyword arguments.
        - name (str): Prefix used for naming the new columns in the DataFrame. This prefix aids in identifying the source of
                      the data and grouping related columns together.
        - step (int): Specifies the interval at which `func` is called across the DataFrame's index when in dynamic mode.
                      It defines how many indices to skip before recalculating the bounds.
        - window_size (int): Defines the number of rows each call to `func` should consider for calculating its returned values.
                             It determines the size of the data window passed to `func` each time it is called.
        - dynamic (bool, optional): If set to False, `func` is called only once and its output is replicated across all rows
                                    of the DataFrame. If True, `func` is invoked repeatedly starting from each `step` interval,
                                    recalculating based on the specified `window_size` until the end of the DataFrame.
        - **func_kwargs: Additional keyword arguments to be passed to `func` each time it is invoked. These should be specified
                         as key-value pairs and will be forwarded directly to `func`.
        
        Returns:
        None: This method modifies the class's DataFrame in-place and does not return any value.
        
        Description:
        - **Non-dynamic Mode**: The method calls `func` once and replicates the result across all rows of the DataFrame, effectively
                                broadcasting the static bounds calculated from the initial segment of the DataFrame defined by
                                `window_size`.
        
        - **Dynamic Mode**: The method iterates through the DataFrame in steps defined by `step`, calling `func` at each step
                            with a slice of the DataFrame defined by `window_size`. This captures the dynamic behavior of the
                            bounds over time, allowing each segment of the DataFrame to have its bounds recalculated and stored,
                            which is particularly useful for time-series analysis where conditions change over time.
        
        Example Usage:
        Assuming an instance `data` with a DataFrame attribute, you might call:
        data.create_bounds(func=data.calculate_moving_average, name='MA', step=5, window_size=20, dynamic=True)
        """
        
        if not dynamic:
            # Call function once to calculate static bounds based on the initial window_size.
            row = func(window_size=window_size, **func_kwargs)
            # Create new columns in the DataFrame using the result and prefix, setting all initial values to NaN.
            for column in row.columns:
                self.df[f'{name}_{column}'] = row[column].iloc[0]
                self.metrics.append(f'{name}_{column}')
            # Fill NaN values forward to propagate the initial calculated values across all rows.
            self.df.fillna(method='ffill', inplace=True)
        
        else:
            # Initialize a DataFrame to store dynamically calculated bounds.
            bounds_df = None
            for i in range(0, len(self.df) - window_size + 1, step):
                # Recalculate bounds at each step and store the results.
                row = func(window_size=window_size, index=i, **func_kwargs)
                if bounds_df is None:
                    # Initialize the dynamic bounds DataFrame using the same index and columns as the main DataFrame.
                    bounds_df = pd.DataFrame(index=self.df.index, columns=row.columns)
                # Insert the calculated row into the dynamic bounds DataFrame.
                bounds_df.loc[row.index[0]] = row.iloc[0]
        
            # Merge the dynamically calculated bounds back into the main DataFrame.
            self.df = pd.concat([self.df, bounds_df], axis=1)
            self.metrics.append(name)
            # Track the metrics or operations performed, helpful for debugging or auditing.
            if all(item[-1].isdigit() for item in bounds_df.columns):
                # Remove the last two characters and duplicates
                metrics_to_add = list(dict.fromkeys([item[:-2] for item in bounds_df.columns]))
                self.metrics.extend(metrics_to_add)
                print('Metrics added:', self.metrics)
            else:
                self.metrics.extend(bounds_df.columns)
            
        
        
    def add_range_marker(self, metric, type='Interior', dynamic=False, threshold=50, bounds=[0,1], ref = 'Close'):
        
        """
        Adds a range marker to the DataFrame based on specified conditions, which can be dynamic or static. This method 
        is designed to aid in identifying trading signals based on the given metric and conditions specified.
    
        Parameters:
        - metric (str): Name of the metric or column in the DataFrame to which the marker will be applied.
        - type (str): Type of marker based on the relationship to the data. Valid types include:
            'Interior': Marks data points within the specified bounds.
            'Exterior': Marks data points outside the specified bounds.
            'Above': Marks data points above the specified threshold.
            'Below': Marks data points below the specified threshold.
        - dynamic (bool, optional): Indicates if the marker adapts to changing data within the DataFrame. 
                                    Defaults to False for static thresholds or bounds.
        - threshold (float, optional): Value used as a threshold for 'Above' or 'Below' types. Default is 50.
        - bounds (list, optional): Two-element list specifying lower and upper bounds for 'Interior' or 'Exterior' types.
                                   Default is [0, 1].
        - ref (str, optional): Reference column against which bounds or thresholds are compared. Default is 'Close'.
    
        Returns:
        None: Modifies the DataFrame in-place by adding a new column to indicate the presence of markers based on 
              the specified conditions.
    
        Example Usage:
        Assuming an instance `trading_data` with a DataFrame containing a 'Close' column, you might call:
        trading_data.add_range_marker(metric='Volume', type='Above', threshold=100)
        This would add a column to indicate where the 'Volume' is above 100.
    
        Notes:
        - This method checks if the specified metric is present in the DataFrame's columns.
        - For dynamic markers, setup for dynamic conditions should be pre-configured in the DataFrame.
        """
        
        if metric not in self.metrics:
            print(f"{metric} is not in metrics. The available metrics are: {self.metrics}")
            return
    
        method = 'bounds' if type in ['Interior', 'Exterior'] else 'threshold'
        reference_column = self.df[ref]
        
        
        if dynamic:
            if method == 'bounds':
                col_name = f"{metric}_{type}_Signal"
                columns = [col for col in self.df.columns if col.startswith(metric + '_') and ('Lower' in col or 'Upper' in col)]
                lower_cols, upper_cols = columns[::2], columns[1::2]
                
                masks = [pd.Series(
                    np.where(
                        self.df[lower].isna() | self.df[upper].isna(),
                        np.nan,  # Set to NaN if either boundary is NaN
                        (reference_column >= self.df[lower]) & (reference_column <= self.df[upper])  # Regular boundary checking
                    ), dtype = 'boolean',  index=self.df.index)
                    for lower, upper in zip(lower_cols, upper_cols)
                ]
                
                #can skipna because combing nans -> False is fine
                combined_mask = pd.concat(masks, axis=1).any(axis=1, skipna=True)
                
                if type == 'Exterior':
                    combined_mask = ~combined_mask
                    
                self.df[col_name] = combined_mask 
                
            elif method == 'threshold':
                # Placeholder for dynamic threshold logic if needed
                columns = [col for col in self.df.columns if col.startswith(metric)]
                col_names = [f"{col}_{type}_Signal" for col in columns]
                print(col_names)
                if type == 'Above':
                    
                    masks = [
                        pd.Series(
                            np.where(
                                self.df[thresh].isna(),  # Check if either comparison value is NaN
                                np.nan,  # Keep NaN in the output where appropriate
                                reference_column >= self.df[thresh]  # Perform the comparison where both values are available
                            ),
                            index=self.df.index,
                            dtype="boolean"  # Ensure dtype is set to handle Boolean with NaN
                        )
                        
                        for thresh in columns
                    ]
                elif type == 'Below':
                    masks = [
                        pd.Series(
                            np.where(
                                self.df[thresh].isna(),  # Check if either comparison value is NaN
                                np.nan,  # Keep NaN in the output where appropriate
                                reference_column <= self.df[thresh]  # Perform the comparison where both values are available
                            ),
                            index=self.df.index,
                            dtype="boolean"  # Ensure dtype is set to handle Boolean with NaN
                        )
                        for thresh in columns
                    ]
                    
                self.df[col_names] = pd.concat(masks, axis=1).set_axis(col_names, axis=1)
                
        else:
            col_name = f"{metric}_{type}_Signal"
            if method == 'bounds':
                condition = (self.df[metric] >= bounds[0]) & (self.df[metric] <= bounds[1]) 
                if type == 'Exterior':
                    condition = ~condition
            else:
                condition = (self.df[metric] > threshold)
                if type == 'Below':
                    condition = ~condition
        
            self.df[col_name] = condition 

            
    def requirement_cross(self, requirement, metric, wait=2, window=10):
        
        """
        Detects changes from False to True in a specified boolean column ('requirement') of the DataFrame, and assesses
        whether the True state is sustained for a defined number of consecutive periods ('wait'). If sustained, it signals
        True for a specified window ('window') of periods.
    
        Parameters:
        - requirement (str): The name of the boolean column in the DataFrame to analyze.
        - metric (str): The metric used to identify specific columns related to 'requirement' if multiple exist.
        - wait (int, optional): The minimum number of consecutive periods the value must remain True to be considered sustained.
                                Defaults to 2.
        - window (int, optional): The number of subsequent periods to signal True after a sustained True condition is detected.
                                  Defaults to 10.
    
        Returns:
        None: Modifies the DataFrame in-place by adding a new column named '{requirement}_Cross' to indicate where the
              signal is True based on the defined criteria.
    
        Detailed Explanation:
        - **Cross Detection**: The method identifies points where the value in the 'requirement' column transitions
                               from False to True.
        - **Sustained True**: It checks if this True state persists for at least 'wait' consecutive periods.
        - **Signal Creation**: If sustained, it triggers a 'True' signal in the 'cross_signal' column for 'window' periods
                               beginning from the point where the True state is first sustained.
    
        Example Usage:
        Assuming an instance `data` with a DataFrame that includes a boolean column 'condition_met', you might call:
        data.requirement_cross(requirement='condition_met', wait=3, window=15)
        This would set a 'condition_met_Cross' column to True for 15 periods following any point where 'condition_met'
        remains True for at least 3 consecutive periods after initially switching from False to True.
    
        Notes:
        - This method requires that the 'requirement' column and the DataFrame already exist.
        - Useful for event-based data analysis or conditions tracking over time.
        - The function dynamically identifies relevant columns based on the 'metric' and 'requirement' to cater to complex
          DataFrame structures with multiple related metrics.
        """
        
        if 'Below' in requirement:
            columns = [col for col in self.df.columns if col.startswith(metric + '_') and ('Below' in col)]
        elif 'Above' in requirement:
            columns = [col for col in self.df.columns if col.startswith(metric + '_') and ('Above' in col)]
        else: # It's either Interior of Exterior
            columns = [col for col in self.df.columns if col.startswith(metric + '_') and ('Interior' in col or 'Exterior' in col)]
        
        # Initialize a Series to hold combined results, starting with all False
        combined_signal = pd.Series(False, index=self.df.index)
        
        # Process each column to detect crossings and sustained conditions
        for column_name in columns:
            column = self.df[column_name]
            
            # Determine where the value changes from False to True
            change_to_true = (column.shift(1) == False) & (column == True)
            
            # Identify sustained True values
            rolling_sum = column.rolling(window=wait, min_periods=1).sum()
            sustained_true = rolling_sum >= wait
            
            # Find the exact point where the condition has been sustained
            first_sustained = sustained_true & change_to_true.shift(wait - 1, fill_value=False)
            
            # Broadcast this sustained condition into the future for 'window' periods
            signal_start = first_sustained.reindex(self.df.index, fill_value=False)
            signal_end = signal_start.shift(window, fill_value=False).reindex(self.df.index, fill_value=False)
            
            # Create a block of True values from start to end
            individual_signal = (signal_start.cumsum() - signal_end.cumsum()) > 0
            
            # Combine the individual signal with the overall combined signal
            combined_signal = combined_signal | individual_signal
        
        # Assign the combined result to a new column in the DataFrame
        self.df[requirement + '_Cross'] = combined_signal
    
        
        
    def show_metrics(self):
        print('Available metrics are: {}'.format(self.metrics))
            
        
    def get_signals(self, independent, meet_requirements, buy_markers_to_use, sell_markers_to_use):
        
        """
        Analyzes specified markers within a DataFrame to generate buy and sell signals based on the confluence
        of these markers and additional requirements. This method allows for configuring how signals are determined,
        either independently or dependently among the markers.
    
        Parameters:
            independent (bool): Determines how the markers are evaluated for generating signals.
                                If True, any single marker meeting the condition can trigger a signal.
                                If False, all markers must meet the condition simultaneously to trigger a signal.
            meet_requirements (list): A list of column names in the DataFrame that must all be True for the
                                      confluence condition to be met. If empty or None, it is assumed that there
                                      are no confluence requirements, and the condition defaults to True for all rows.
            markers_to_use (list): A list of column names in the DataFrame representing the markers to be evaluated
                                   for buying or selling signals.
    
        Processing Steps:
            1. **Confluence Evaluation**: Determines a 'confluence' column based on the `meet_requirements` parameter.
               If `meet_requirements` is not provided or is empty, every row's 'confluence' value is set to True.
               Otherwise, 'confluence' is True only where all specified `meet_requirements` are True.
            2. **Signal Detection**: Depending on the `independent` parameter, the method checks the `markers_to_use`
               to identify rows where buy or sell conditions are met.
               - For buying signals, checks if markers are equal to 1.
               - For selling signals, checks if markers are equal to -1.
               The conjunction with the 'confluence' column determines the final signal.
    
        Returns:
            None: This method modifies the DataFrame in-place, adding two new DataFrames to the instance:
                  `buy_df` and `sell_df`, which hold the 'Close' prices where buy and sell conditions, respectively,
                  were met based on the evaluated conditions.
    
        Example Usage:
            Assume `trading_data` is an instance with a DataFrame that includes the necessary markers and a 'Close' price column. 
            The method might be called as follows:
            trading_data.get_signals(independent=False,
                                     meet_requirements=['cond1', 'cond2'],
                                     markers_to_use=['marker1', 'marker2'])
    
            This configuration requires that all specified markers ('marker1', 'marker2') agree and that both 'cond1'
            and 'cond2' are True to trigger buying or selling signals.
    
        Notes:
            - It is important that all columns referenced in `meet_requirements` and `markers_to_use` exist in the DataFrame.
            - The method assumes that marker values are set such that 1 indicates a potential buy signal and -1 indicates a sell signal.
        """
        
        # If meet_requirements is not provided or is empty, default the confluence condition to True
        if not meet_requirements:
            self.df['confluence'] = True
        else:
            # Only set confluence to True where all specified requirements are met
            self.df['confluence'] = self.df[meet_requirements].all(axis=1)
        
        # Set conditions for buying and selling based on the independence of markers
        if not independent:
            # Require all markers to indicate a buy (1) or sell (-1) while also meeting confluence requirements
            buy_condition = (self.df[buy_markers_to_use] == True).all(axis=1) & self.df['confluence']
            sell_condition = (self.df[sell_markers_to_use] == True).all(axis=1) & self.df['confluence']
        else:
            # Any marker can indicate a buy or sell while meeting confluence requirements
            buy_condition = (self.df[buy_markers_to_use] == True).any(axis=1) & self.df['confluence']
            sell_condition = (self.df[sell_markers_to_use] == True).any(axis=1) & self.df['confluence']
        
        self.markers_to_use = [buy_markers_to_use, sell_markers_to_use]
        self.independent = independent
        
        # Create a unified signal column where -1 is sell, 0 is neutral, 1 is buy
        self.df['Signal'] = 0  # Default to neutral
        self.df.loc[buy_condition, 'Signal'] = 1
        self.df.loc[sell_condition, 'Signal'] = -1

        # Optionally, you can display buy and sell signals
        print("Buy Signals:\n", self.df[self.df['Signal'] == 1]['Close'])
        print("Sell Signals:\n", self.df[self.df['Signal'] == -1]['Close'])
        
    
    def get_returns_histogram(self, window_size=30, bins=50):
    
        # Calculate the percentage returns for each window
        returns = (self.df['Close'].pct_change() * 100)
            
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        ax.hist(returns[-window_size:], bins = bins)
        

    def get_volume_profile(self, window_size=1, bins=100, index=1, prom_factor=0.1, show=True):
        
        """
        Calculates and optionally displays a volume profile using kernel density estimation (KDE) for a specified
        window of trade data. It identifies key peaks (POCs) and their properties, and optionally plots the volume
        profile with peaks and their widths.
    
        Parameters:
            window_size (int): The size of the window (in terms of number of data points) to consider for the profile.
                               Typically, this would be the number of trades or bars to include.
            bins (int): Number of bins for histogramming the KDE output. This parameter is currently not in use but
                        reserved for future implementations or adjustments.
            index (int): The starting index from which to calculate the profile in the DataFrame.
            prom_factor (float): The prominence factor used to determine the significance of peaks in the KDE.
                                 Peaks with a prominence greater than `prom_factor` times the maximum of the KDE are considered.
            show (bool): Whether to show a plot of the volume profile with peaks and their widths.
    
        Returns:
            pandas.DataFrame: A DataFrame containing the Points of Control (POCs) with their corresponding
                              price, prominence, density, and upper and lower bounds.
    
        Processing Steps:
            1. Calculate the KDE for the 'Close' prices weighted by 'Volume'.
            2. Identify peaks in the KDE with prominence defined by `prom_factor`.
            3. Calculate the widths at each peak to determine the range of influence for each POC.
            4. If `show` is True, plot the volume profile with peaks and their widths.
            5. Store data about each POC including its price level, prominence, and density.
    
        Example Usage:
            Assuming an instance `market_data` with a DataFrame containing 'Close' and 'Volume' columns, you might call:
            poc_data = market_data.get_volume_profile(window_size=30, index=100, prom_factor=0.2, show=True)
            This would calculate and possibly display the volume profile for 30 data points starting at index 100,
            considering peaks significant if they have at least 20% of the maximum KDE value.
    
        Notes:
            - The method requires that the DataFrame (`self.df`) contains 'Close' and 'Volume' columns.
            - The plot output is configured with basic styling; customization may require modifications to the plotting block.
        """
        
        close = self.df['Close'][index:index + window_size]
        volume = self.df['Volume'][index:index + window_size]
        kde_factor = 0.05
        num_samples = 500
        kde = stats.gaussian_kde(close, weights=volume, bw_method=kde_factor)
        xr = np.linspace(close.min(), close.max(), num_samples)
        kdy = kde(xr)
        ticks_per_sample = (xr.max() - xr.min()) / num_samples
    
        # Find peaks with the specified prominence
        min_prom = kdy.max() * prom_factor
        peaks, peak_props = signal.find_peaks(kdy, prominence=min_prom, width=1)
        pkx = xr[peaks]
        pky = kdy[peaks]
    
        # Calculate widths of peaks
        left_ips = peak_props['left_ips']
        right_ips = peak_props['right_ips']
        width_x0 = xr.min() + (left_ips * ticks_per_sample)
        width_x1 = xr.min() + (right_ips * ticks_per_sample)
        width_y = peak_props['width_heights']
    
        # Optionally show the plot
        if show:
            plt.figure(figsize=(10, 5), dpi=50)
            plt.grid()
            plt.plot(xr, kdy, label='KDE of Volume Profile')
            plt.plot(pkx, pky, "x", label='Peaks')
            for i in range(len(width_x0)):
                plt.hlines(width_y[i], width_x0[i], width_x1[i], color='red', label='Width' if i == 0 else "")
            plt.title('Volume Profile with Peaks and Widths - ' + str(close.index[-1]))
            plt.xlabel('Price')
            plt.ylabel('Density')
            plt.legend()
            plt.show()
            plt.close()
    
        # Collect POC data
        self.pocs = []
        for x, y0, y1, x0, x1, y, d in zip(pkx, pky, pky - peak_props['prominences'],
                                           width_x0, width_x1, width_y,
                                           [kde.integrate_box_1d(x0, x1) for x0, x1 in zip(width_x0, width_x1)]):
            poc = {'poc': x, 'prominence': y1 - y0, 'xL': x0, 'xR': x1, 'density': d}
            self.pocs.append(poc)
    
        self.pocs.sort(key=lambda p: p['poc'])
        df_data = {}
        group = 'VPF'
        for i, poc in enumerate(self.pocs):
            df_data[f'{group}_POC_{i + 1}'] = poc['poc']
            df_data[f'{group}_Lower_{i + 1}'] = poc['xL']
            df_data[f'{group}_Upper_{i + 1}'] = poc['xR']
            df_data[f'{group}_Prominence_{i + 1}'] = poc['prominence']
            df_data[f'{group}_Density_{i + 1}'] = poc['density']
    
        final_index = close.index[-1]
        poc_df = pd.DataFrame(df_data, index=[final_index])
        return poc_df

    def vsa_indicator(self, norm_lookback = 168):
        """
        Computes the Volume Spread Analysis (VSA) indicator for the DataFrame. This indicator evaluates the relationship
        between price range and volume, providing insights into potential market strength or weakness based on volume spread.
    
        The method calculates normalized price range and volume, then uses linear regression over a rolling window to 
        estimate expected price range based on volume. The deviation of the actual range from this expected range provides
        the VSA indicator values.
    
        Parameters:
        - norm_lookback (int, optional): The number of periods used to calculate the Average True Range (ATR) for normalization
                                         and the median volume. This also determines the size of the rolling window for the 
                                         regression analysis. Defaults to 168 periods.
    
        Returns:
        - numpy.ndarray: An array of VSA indicator values where each value represents the deviation of the actual normalized 
                         price range from the expected range calculated via regression. The array matches the length of the
                         DataFrame with NaN values for the initial periods where the calculation cannot be performed.
    
        Detailed Explanation:
        - **Normalization**: The method first normalizes the price range by the ATR and the volume by its median over the 
                             specified lookback period.
        - **Regression Analysis**: For each point from twice the lookback period onwards, it performs a linear regression 
                                   between the normalized volume and the normalized range over the lookback period. This
                                   regression estimates the expected normalized range from the current volume.
        - **Deviation Calculation**: The deviation of the actual normalized range from the expected range (based on the 
                                     regression) is calculated. Positive values indicate a larger actual range compared to 
                                     expected (suggesting market strength), while negative values suggest a smaller range 
                                     than expected (potentially indicating market weakness).
        
        Notes:
        - This indicator is typically used in trading to identify periods where price movements are not justified by volume,
          suggesting potential reversal points or continuation signals.
        - Ensure the DataFrame (`self.df`) includes the necessary columns: 'High', 'Low', 'Close', and 'Volume'.
        """
        
        data = self.df.copy()
        
        atr = ta.ATR(self.df['High'], self.df['Low'], self.df['Close'], norm_lookback)
        vol_med = self.df['Volume'].rolling(norm_lookback).median()
    
        data['norm_range'] = abs(self.df['High'] - self.df['Low']) / atr 
        data['norm_volume'] = self.df['Volume'] / vol_med 
    
        norm_vol = data['norm_volume'].to_numpy()
        norm_range = data['norm_range'].to_numpy()
    
        range_dev = np.zeros(len(self.df))
        range_dev[:] = np.nan
    
        for i in range(norm_lookback * 2, len(data)):
            window = data.iloc[i - norm_lookback + 1: i+ 1]
            slope, intercept, r_val,_,_ = stats.linregress(window['norm_volume'], window['norm_range'])
    
            if slope <= 0.0 or r_val < 0.2:
                range_dev[i] = 0.0
                continue
           
            pred_range = intercept + slope * norm_vol[i]
            range_dev[i] = norm_range[i] - pred_range
           
        return range_dev
        
                
    def show(self, show_volume=False, points=0, add_line=True, lines=['MA_30'], show_pocs=True, use_densities=True):
        
        """
        Generates an interactive plot of trading data with various overlays including candlesticks,
        volume bars, points of control (POCs), and buy/sell signals. This method leverages Plotly for
        visualization, providing a dynamic and informative charting interface.
    
        Parameters:
            show_volume (bool): If True, volume bars are displayed beneath the candlesticks.
            points (int): The number of data points from the end of the dataset to display. If 0, all data points are displayed.
            add_line (bool): If True, additional technical analysis lines specified in the `lines` parameter are added to the plot.
            lines (list of str): A list of column names from the DataFrame which contain data for additional lines to be plotted, 
                                 such as moving averages or other indicators.
            show_pocs (bool): If True, points of control (POCs) derived from volume profile analysis are plotted.
            use_densities (bool): Not currently used in the method, but reserved for future functionality to toggle density-based visualizations.
    
        Processing Steps:
            1. **Candlestick Chart**: Creates a candlestick chart for the price movement.
            2. **Volume Bars**: Optionally adds volume bars below the candlesticks if `show_volume` is True.
            3. **POCs Visualization**: If `show_pocs` is True, plots markers for POCs, upper and lower bounds from volume profile analysis.
            4. **Trading Signals**: Marks buy and sell signals with up and down triangles on the chart.
            5. **Additional Lines**: Plots any additional technical analysis lines (like moving averages) specified in the `lines` parameter.
    
        Returns:
            None: This method directly renders a Plotly interactive figure and does not return a value.
    
        Example Usage:
            Assuming an instance `trading_data` with the necessary data prepared, you might call:
            trading_data.show(show_volume=True, points=100, add_line=True, lines=['MA_50', 'EMA_20'])
    
        Notes:
            - This method assumes the presence of 'Open', 'High', 'Low', 'Close', and optionally 'Volume' columns in the DataFrame.
            - It also utilizes columns for buy and sell signals, and expects these to be present as set by prior analysis methods.
            - Proper functionality requires that all referenced columns (`lines`, etc.) exist in the DataFrame.
        """
        
        trunc_df = self.df.iloc[-points:, :] if points else self.df
        # Create candlestick trace
        candlesticks = go.Candlestick(
            x=trunc_df.index,
            open=trunc_df['Open'],
            high=trunc_df['High'],
            low=trunc_df['Low'],
            close=trunc_df['Close'],
            showlegend=False
        )
    
        # Setup figure with subplots
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(candlesticks, secondary_y=True)
    
        # Handle POCs display
        if show_pocs:
            pocs = trunc_df.filter(like='_POC', axis=1)
            uppers = trunc_df.filter(like='_Upper', axis=1)
            lowers = trunc_df.filter(like='_Lower', axis=1)
            # Display each POC, upper, and lower trace
            for i in range(pocs.shape[1]):
                fig.add_trace(go.Scatter(
                    x=pocs.iloc[:, i].index,
                    y=pocs.iloc[:, i].values,
                    mode='markers',
                    marker=dict(color='Gray', size=1),
                    name=pocs.columns[i]
                ), secondary_y=True)
                fig.add_trace(go.Scatter(
                    x=uppers.iloc[:, i].index,
                    y=uppers.iloc[:, i].values,
                    mode='markers',
                    marker=dict(color='Green', size=1),
                    name=uppers.columns[i]
                ), secondary_y=True)
                fig.add_trace(go.Scatter(
                    x=lowers.iloc[:, i].index,
                    y=lowers.iloc[:, i].values,
                    mode='markers',
                    marker=dict(color='Red', size=1),
                    name=lowers.columns[i]
                ), secondary_y=True)
    
        # Optionally add volume bars
        if show_volume:
            fig.add_trace(go.Bar(
                x=trunc_df.index,
                y=trunc_df['Volume'],
                showlegend=False,
                marker={"color": "rgba(128,128,128,0.7)"}
            ), secondary_y=False)
            fig.update_yaxes(title="Volume", secondary_y=False, showgrid=False)
    
        # Add traces for buy and sell signals
        fig.add_trace(go.Scatter(
            x=self.df[self.df['Signal'] == 1]['Close'].index,
            y=self.df[self.df['Signal'] == 1]['Close'].values * 0.998,  # slightly below the close price for visibility
            mode='markers',
            marker=dict(color='blue', symbol='triangle-up'),
            name='Buy Signal'
        ), secondary_y=True)
        
        fig.add_trace(go.Scatter(
            x=self.df[self.df['Signal'] == -1]['Close'].index,
            y=self.df[self.df['Signal'] == -1]['Close'].values * 1.002,  # slightly above the close price for visibility
            mode='markers',
            marker=dict(color='orange', symbol='triangle-down'),
            name='Sell Signal'
        ), secondary_y=True)
    
        # Add additional TA lines if specified
        if add_line:
            for line in lines:
                if line in trunc_df.columns:
                    fig.add_trace(go.Scatter(
                        x=trunc_df.index,
                        y=trunc_df[line],
                        name=line,
                        showlegend=True
                    ), secondary_y=True)
    
        # Finalize figure layout
        fig.update_layout(
            title=f'{self.symbol} - {self.interval}, Markers: {self.markers_to_use}, Independent = {self.independent}',
            height=800,
            xaxis={"rangeslider": {"visible": True}},
        )
        fig.update_yaxes(title="Price $", secondary_y=True, showgrid=True, fixedrange=False)
        fig.show()



class BackTester:
    
    """
    A class to backtest trading strategies based on entry and exit signals provided within a DataFrame. This tester
    accounts for commissions, stop loss, take profit, and the option to use Average True Range (ATR) for dynamic
    loss/profit settings.

    Attributes:
    - data (pd.DataFrame): The data used for backtesting, which must include 'High', 'Low', 'Close', and 'Signal' columns.
    - commission (float): The commission percentage per trade.
    - stoploss (float): The stop loss percentage (relative to entry price).
    - takeprofit (float): The take profit percentage (relative to entry price).
    - use_atr (bool): Flag to determine whether to use ATR for stop loss and take profit calculation.
    - atr_multiplier (float): Multiplier to scale the ATR value when use_atr is True.
    - trailing (bool): Flag to enable trailing stop losses.
    - capital (float): Initial capital for backtesting.
    - atr (pd.Series): Series of ATR values computed from data if use_atr is True.

    Methods:
    - run_strategy: Executes the trading strategy across the provided data.
    - open_position: Opens a new trading position based on the signal.
    - update_trailing_stop: Updates the stop loss value for trailing stops.
    - close_position: Closes an open trading position.
    - calculate_metrics: Calculates and prints out key trading metrics.
    - plot: Plots the backtest results including capital over time and price movements.
    """
    
    def __init__(self, data, commission=0.002, stoploss=0.01, takeprofit=0.02, use_atr=False, atr_multiplier=1.0, trailing=False, capital = 1000):
        
        """
        Initializes the BackTester object with trading parameters and initial conditions.

        Parameters:
        - data (pd.DataFrame): Input data for backtesting.
        - commission (float): Commission per trade as a fraction (default 0.002 or 0.2%).
        - stoploss (float): Stop loss threshold as a fraction of entry price (default 0.01 or 1%).
        - takeprofit (float): Take profit threshold as a fraction of entry price (default 0.02 or 2%).
        - use_atr (bool): If True, use ATR to set stop loss and take profit dynamically (default False).
        - atr_multiplier (float): Multiplier for ATR when setting dynamic stops or profits (default 1.0).
        - trailing (bool): If True, use trailing stop losses (default False).
        - capital (float): Initial trading capital (default 1000).
        """
        
        self.data = data
        self.commission = commission
        self.stoploss = stoploss
        self.takeprofit = takeprofit
        self.use_atr = use_atr
        self.atr_multiplier = atr_multiplier
        self.trailing = trailing
        self.capital = capital  # Initial capital
        self.position = 0  # 0 = no position, 1 = long, -1 = short
        self.entry_price = None
        self.stop_loss_price = None
        self.take_profit_price = None
        self.trade_log = pd.DataFrame(columns=['Timestamp', 'Price', 'Action', 'Profit/Loss', 'Commission', 'Cumulative Profit/Loss'])
        self.atr = ta.ATR(self.data['High'], self.data['Low'], self.data['Close'], 14) if self.use_atr else None
        
        
    def run_strategy(self):
        
        """
        Executes the trading strategy by iterating through the DataFrame and managing positions based on 'Signal' column
        and price movements. Records all trades in the trade_log attribute.
        """
        
        for index, row in self.data.iterrows():
            current_time = index
            current_price = row['Close']
            signal = row['Signal']
            self.atr_value = self.atr.loc[current_time] 
            
            if self.position != 0 and self.entry_price is not None:
                if self.trailing:
                    self.update_trailing_stop(current_price, current_time)
                
                # Check stop loss and take profit conditions
                if self.position == 1:
                    if current_price <= self.stop_loss_price:
                        self.close_position(current_price, 'hit stop loss', current_time)
                        continue  # Skip to next iteration after closing position
                    elif not self.trailing and self.take_profit_price is not None and current_price >= self.take_profit_price:
                        self.close_position(current_price, 'hit take profit', current_time)
                        continue  # Skip to next iteration after closing position
                elif self.position == -1:
                    if current_price >= self.stop_loss_price:
                        self.close_position(current_price, 'hit stop loss', current_time)
                        continue  # Skip to next iteration after closing position
                    elif not self.trailing and self.take_profit_price is not None and current_price <= self.take_profit_price:
                        self.close_position(current_price, 'hit take profit', current_time)
                        continue  # Skip to next iteration after closing position
    
            # Open new positions or switch positions based on signals
            if signal == 1 and self.position != 1:
                if self.position == -1:
                    self.close_position(current_price, 'switching position', current_time)
                self.open_position(current_price, 'long', current_time)
            elif signal == -1 and self.position != -1:
                if self.position == 1:
                    self.close_position(current_price, 'switching position', current_time)
                self.open_position(current_price, 'short', current_time)


    def open_position(self, price, type, current_time):
        
        """
        Opens a new position either 'long' or 'short' based on the type specified, adjusting capital for commission and
        setting up stop loss and take profit levels.

        Parameters:
        - price (float): The entry price for the position.
        - type (str): 'long' or 'short', indicating the type of position to open.
        - current_time (timestamp): The timestamp at which the position is being opened.
        """
        
        self.entry_price = price
        self.position = 1 if type == 'long' else -1
        self.entry_commission = self.capital * self.commission
        self.capital -= self.entry_commission
        
        if self.use_atr:
            self.stop_loss_price = price - (self.atr_value * self.atr_multiplier) if type == 'long' else price + (self.atr_value * self.atr_multiplier)
            if not self.trailing:
                self.take_profit_price = price + (self.atr_value * self.atr_multiplier ) if type == 'long' else price - (self.atr_value * self.atr_multiplier)
        else:
            self.stop_loss_price = price * (1 - self.stoploss) if type == 'long' else price * (1 + self.stoploss)
            if not self.trailing:
                self.take_profit_price = price * (1 + self.takeprofit) if type == 'long' else price * (1 - self.takeprofit)
        
        if self.trailing:
            self.take_profit_price = None  # Ensure take profit is disabled when trailing
        
        n = len(self.trade_log)
        # Append to DataFrame
        self.trade_log = pd.concat([self.trade_log, pd.DataFrame({
            'Timestamp': [current_time],
            'Price': [price],
            'Action': [f"open {type}"],
            'Profit/Loss': [-self.entry_commission],
            'Commission': [self.entry_commission],
            'Cumulative Profit/Loss': [np.nan], 
            'Capital': [self.capital]
        })], ignore_index=True)
        
        self.trade_log.loc[n,'Cumulative Profit/Loss'] = self.trade_log['Profit/Loss'].dropna().sum(), 
        

    def close_position(self, price, reason, current_time):
        
        """
        Closes an open trading position, calculating profit/loss and adjusting capital accordingly.

        Parameters:
        - price (float): The price at which the position is closed.
        - reason (str): The reason for closing the position ('hit stop loss', 'hit take profit', 'switching position').
        - current_time (timestamp): The timestamp at which the position is being closed.
        """
        
        if self.entry_price is not None and self.position != 0:
            if self.position == 1:
                if reason == 'hit stop loss':
                    self.capital = self.capital * (self.stop_loss_price/self.entry_price)
                    price = self.stop_loss_price
                elif reason == 'hit take profit':
                    self.capital = self.capital * (self.take_profit_price/self.entry_price)
                    price = self.take_profit_price
                elif reason == 'switching position':
                    self.capital = self.capital * (price/self.entry_price)
            elif self.position == -1:
                if reason == 'hit stop loss':
                    self.capital = self.capital * (self.entry_price/self.stop_loss_price)
                    price = self.stop_loss_price
                elif reason == 'hit take profit':
                    self.capital = self.capital * (self.entry_price/self.take_profit_price)
                    price = self.take_profit_price
                elif reason == 'switching position':
                    self.capital = self.capital * (self.entry_price/price)
                    
            self.exit_commission = (self.capital) * self.commission
            n = len(self.trade_log)
            profit = (self.capital - self.trade_log.loc[n-1,'Capital'])

            new_row = pd.DataFrame({
                'Timestamp': [current_time],
                'Price': [price],
                'Action': [f"close ({reason})"],
                'Profit/Loss': [profit],
                'Commission': [self.exit_commission],
                'Cumulative Profit/Loss': [np.nan],
                'Capital': [self.capital]
            })
            self.trade_log = pd.concat([self.trade_log, new_row], ignore_index=True)
            self.trade_log.loc[n,'Cumulative Profit/Loss'] = self.trade_log['Profit/Loss'].dropna().sum(), 
            self.entry_price = None
            self.position = 0
            self.stop_loss_price = None
            self.take_profit_price = None


    def update_trailing_stop(self, current_price, current_time):
        
        """
        Updates the stop loss value for a trailing stop, adjusting the stop level as the price moves favorably.

        Parameters:
        - current_price (float): The current market price to update the trailing stop against.
        - current_time (timestamp): The current timestamp for updating the trailing stop.
        """
        
        if self.position == 1 and self.stop_loss_price is not None:
            if self.use_atr:
                self.stop_loss_price = max(self.stop_loss_price, current_price - (self.atr_value * self.atr_multiplier))
            else:
                self.stop_loss_price = max(self.stop_loss_price, current_price * (1-self.stoploss))
            
        elif self.position == -1 and self.stop_loss_price is not None:
            if self.use_atr:
                self.stop_loss_price = min(self.stop_loss_price, current_price + (self.atr_value * self.atr_multiplier))
            else:
                self.stop_loss_price = max(self.stop_loss_price, current_price * (1+self.stoploss))


    def calculate_metrics(self, show_trade_log=False):
        
        """
        Calculates and prints trading performance metrics such as total profit, win rate, and largest wins/losses.

        Parameters:
        - show_trade_log (bool): If True, also prints the trade log. Default is False.
        """
        
        profits = self.trade_log['Profit/Loss'].dropna()
        total_profit = profits.sum()
        win_trades = profits[profits > 0]
        win_rate = len(win_trades) / len(profits) if not profits.empty else 0
        largest_win_money = self.trade_log['Profit/Loss'].max()
        maxidx = self.trade_log['Profit/Loss'].idxmax()
        largest_win_percent =  (largest_win_money / self.trade_log.loc[maxidx-1,'Capital'])* 100
        largest_loss_money = self.trade_log['Profit/Loss'].min()
        minidx = self.trade_log['Profit/Loss'].idxmin()
        largest_loss_percent = (largest_loss_money / self.trade_log.loc[minidx-1,'Capital'])* 100
        
        metrics = {
            'Total Profit': f'£{total_profit:.2f}',
            'Final Capital': f'£{self.capital:.2f}',
            'Number of Trades': len(profits),
            'Win Rate': f"{win_rate:.2%}",
            'Average Profit per Trade': f'£{total_profit / len(profits) if not profits.empty else 0:.2f}',
            'Largest Win': f'£{largest_win_money:.2f} = {largest_win_percent:.2f}%',
            'Largest Loss': f'£{largest_loss_money:.2f} = {largest_loss_percent:.2f}%'
        }

        if show_trade_log:
            print("\nTrade Log:")
            print(self.trade_log.to_string(index=False))

        print("\nTrade Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value}")


    def plot(self):
        
        """
        Plots the results of the backtest including the price data and capital evolution over time on a graph.

        Uses matplotlib for plotting, displaying key trading actions and the evolution of capital.
        """
        
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        ax.tick_params(axis='y', labelcolor='black')  

        ax.grid()
        ax.set_ylabel("Close Price", fontsize = 15)
        ax.set_xlabel("Time", fontsize = 15)
        ax.plot(self.data.index, self.data['Close'], color = 'black')

        ax2 = ax.twinx()
        ax2.tick_params(axis='y', labelcolor='orange')
        ax2.set_ylabel("Capital", fontsize = 15)

        ax2.plot(self.trade_log['Timestamp'], self.trade_log['Capital'], color = 'orange')
    

      

    
# Get historical data
data = Data()
data.symbol = 'BTCUSDT'
data.interval = '5m'
data.start_str = '2024-01-01' #YYYY-MM-DD

data.get_historical_klines(show_dataframe = True)


#%%
# Here, dynamic volume profile bounds are being created. The `create_bounds` method uses a function `get_volume_profile`,
# which is partially applied with specific parameters like bins, prom_factor, and show.
# - `dynamic=True` indicates the bounds are recalculated dynamically for each step.
# - `step=1` implies the bounds are updated for every new row in the data.
# - `window_size=500` specifies the number of rows in each sliding window over which the volume profile is calculated.
# - `name='VPF'` names the resulting columns prefixed with 'VPF' for identification.
# - `func=partial(...)` specifies the function to calculate bounds with the indicated parameters.
data.create_bounds(dynamic=True, step=1, window_size=500, name='VPF', func=partial(data.get_volume_profile, bins=50, prom_factor=0.5, show=False))

# Adding range markers for volume profile upper and lower bounds. These indicate where the 'Close' price is above the upper bound or below the lower bound.
# - 'VPF_Upper' and 'VPF_Lower' are column names derived from the 'VPF' prefix set in the create_bounds method.
# - `type='Above'` and `type='Below'` specify the type of marker condition (above or below the specified threshold or bound).
# - `dynamic=True` implies the range conditions are recalculated dynamically as new data comes in.
# - `ref='Close'` indicates the reference column against which the bounds are compared.
data.add_range_marker('VPF_Upper', type='Above', dynamic=True, ref='Close')
data.add_range_marker('VPF_Lower', type='Below', dynamic=True, ref='Close')

# Creating requirements that check for crosses above and below signals based on the VPF range markers.
# - The `requirement_cross` method is used to determine when a marker signal changes from False to True and sustains for a specified time.
# - 'VPF_Below_Signal' and 'VPF_Above_Signal' are generated names based on previous `add_range_marker` calls.
# - `metric='VPF'` specifies the general metric or basis for these checks.
# - `wait=3` indicates that the True condition must hold for 3 consecutive data points to validate the condition.
# - `window=2` defines how long the signal remains True after the condition is met.
data.requirement_cross('VPF_Below_Signal', metric='VPF', wait=3, window=2)
data.requirement_cross('VPF_Above_Signal', metric='VPF', wait=3, window=2)

# Adding a new indicator, the Relative Strength Index (RSI) for 14 periods to the dataset.
data.add('RSI_14', ta.RSI(data.df['Close'], timeperiod=14))

# Adding range markers for the RSI where signals are generated when RSI crosses below 30 or above 70.
# These are typical thresholds for identifying overbought and oversold conditions in the market.
data.add_range_marker('RSI_14', type='Below', dynamic=False, threshold=30)
data.add_range_marker('RSI_14', type='Above', dynamic=False, threshold=70)

# Adding the Volume Spread Analysis (VSA) indicator to the dataset.
data.add('VSA', data.vsa_indicator())

# Creating a range marker for the VSA indicator where a signal is generated if the VSA value is above 2.
data.add_range_marker('VSA', type='Above', dynamic=False, threshold=2)

# Generating buy and sell signals based on the previously created indicators and range markers.
# - `independent=False` indicates that the signals depend on the combination of several conditions.
# - `meet_requirements` specifies which conditions must be met for a signal.
# - `buy_markers_to_use` and `sell_markers_to_use` define specific signals that trigger buy or sell actions.
data.get_signals(independent=False, meet_requirements=['VSA_Above_Signal'], buy_markers_to_use=['RSI_14_Below_Signal'], sell_markers_to_use=['RSI_14_Above_Signal'])

#%%

backtester = BackTester(data.df, trailing=True, use_atr = True, commission = 0.002, atr_multiplier = 0.5 )  
backtester.run_strategy()
backtester.calculate_metrics(show_trade_log = True)
backtester.plot()
#%%
data.show(show_volume = True, add_line = False, lines = [], show_pocs = True, use_densities = True)
