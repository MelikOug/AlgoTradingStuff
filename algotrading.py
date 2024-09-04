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
        This method can operate in a static (non-dynamic) or dynamic mode.
    
        Parameters:
        - func (callable): A function that when called returns a pandas DataFrame. This function is expected to
                           take at least a `window_size` parameter and can accept additional keyword arguments.
        - name (str): The prefix used for naming the new columns in the DataFrame. This helps in identifying
                      the source of the data and grouping related columns together.
        - step (int): The interval at which the `func` function is called across the DataFrame's index in dynamic mode.
                      It determines how many indices to skip before recalculating the bounds.
        - window_size (int): The number of rows each call to `func` should consider for calculating its returned values.
                             It specifies the size of the data window passed to `func` each time it is called.
        - dynamic (bool, optional): Determines the mode of operation. If False, the function `func` is called only once,
                                    and its output is replicated across all rows of the DataFrame. If True, `func` is
                                    called repeatedly starting from each `step` interval, recalculating based on the
                                    specified `window_size` until the end of the DataFrame.
        - **func_kwargs: Additional keyword arguments to be passed to `func` each time it is called. These arguments
                         should be specified as key-value pairs, and they will be forwarded directly to the `func`.
    
        Returns:
        None: This method modifies the class's DataFrame in-place and does not return any value.
    
        Description:
        - **Non-dynamic Mode**:
          Calls `func` once and replicates the result across all rows of the DataFrame, effectively broadcasting 
          the static bounds calculated from the initial segment of the DataFrame defined by `window_size`.
        
        - **Dynamic Mode**:
          Iterates through the DataFrame in steps defined by `step`, calling `func` at each step with a slice of
          the DataFrame defined by `window_size`. It captures the dynamic behavior of the bounds over time, allowing
          each segment of the DataFrame to have its bounds recalculated and stored. This is useful for time-series
          analysis where conditions change over time.
    
        Example Usage:
        Assuming an instance `data` with a DataFrame attribute, you might call:
        data.create_bounds(func=data.calculate_moving_average, name='MA', step=5, window_size=20, dynamic=True)
        """
        
        if not dynamic:
            # Retrieve data using the provided function and arguments.
            row = func(window_size=window_size, **func_kwargs)
    
            # Append new columns to the DataFrame with the results, initially setting all values to NaN.
            for column in row.columns:
                self.df[f'{name}_{column}'] = row[column].iloc[0]
    
            # Forward fill the NaN values with the initial calculated value.
            self.df.fillna(method='ffill', inplace=True)
        
        else:
            # Initialize an empty DataFrame for dynamically calculated bounds.
            bounds_df = None
            for i in range(0, len(self.df) - window_size, step):
                # Recalculate bounds at each step and store the results.
                row = func(window_size=window_size, index=i, **func_kwargs)
                if i == 0:
                    # Initialize the DataFrame to store dynamic bounds using the same index and columns as the main DataFrame.
                    bounds_df = pd.DataFrame(index=self.df.index, columns=row.columns)
                # Insert the calculated row into the dynamic bounds DataFrame.
                bounds_df.loc[row.index[0]] = row.iloc[0]
    
            # Merge the dynamically calculated bounds back into the main DataFrame.
            self.df = pd.concat([self.df, bounds_df], axis=1)
            # Track the metrics or operations performed, helpful for debugging or auditing.
            self.metrics.append(name)
            print(self.metrics)

        
        
    def add_range_marker(self, metric, type='Interior', action='long', dynamic=False, requirement=False, threshold=50, bounds=[0,1]):
        
        """
        Adds a range marker to the DataFrame based on specified conditions. This method is used to help identify trading signals
        within the DataFrame based on the metrics provided. The markers can be dynamic or static and can indicate the presence or
        an action based on the trading scenario.
    
        Parameters:
            metric (str): The metric to apply the marker to. This metric should be a column name or a derivative metric already calculated
                          and stored within the DataFrame.
            type (str): The type of marker, which defines how the marking is applied relative to the data. Valid types include:
                        'Interior', 'Exterior', 'Above', 'Below'. 'Interior' and 'Exterior' are used with `bounds` to mark data points
                        within or outside a given range, respectively. 'Above' and 'Below' are used with `threshold` to mark data points
                        above or below a given value.
            action (str): The trading action associated with the marker. Values can be 'long' or 'short', where 'long' typically indicates
                          buying or bullish signals, and 'short' indicates selling or bearish signals.
            dynamic (bool): Specifies whether the marker should consider dynamic boundaries or thresholds that might change over time within
                            the DataFrame. If False, the marker uses static bounds or a static threshold.
            requirement (bool): Determines whether the marker should just indicate presence (True/False) or should indicate an action (action
                                value). If True, the marker will be a boolean indicating whether the condition is met. If False, the marker will
                                be set to `action_value` (1 or -1) where conditions are met, and 0 elsewhere.
            threshold (float): The threshold value used for 'Above' or 'Below' conditions. This parameter defines the critical value at which
                               the marker is applied for these conditions.
            bounds (list): A list of two numbers defining the lower and upper bounds used for 'Interior' or 'Exterior' conditions. This parameter
                           specifies the range within which (or outside of which) the marker should be applied for these conditions.
    
        Returns:
            None: Modifies the DataFrame in-place by adding a new column to indicate the markers based on the specified conditions.
    
        Example Usage:
            Assuming an instance `trading_data` has a DataFrame with a column 'Price', you might call:
            trading_data.add_range_marker(metric='Price', type='Above', action='long', threshold=100)
            This will add a column indicating long action where the 'Price' is above 100.
    
        Notes:
            The method checks if the specified metric is available in the class's list of metrics (`self.metrics`). If not, it returns an error.
            For dynamic calculations, the method requires setting up relevant dynamic calculations before calling this method.
        """
        
        if metric not in self.metrics:
            print(f"{metric} is not in metrics. The available metrics are: {self.metrics}")
            return
    
        action_value = 1 if action == 'long' else -1
        method = 'bounds' if type in ['Interior', 'Exterior'] else 'threshold'
        col_name = f"{metric}_{type}_Signal"
        print(col_name)
        
        if dynamic:
            if method == 'bounds':
                columns = [col for col in self.df.columns if col.startswith(metric + '_') and ('Lower' in col or 'Upper' in col)]
                lower_cols, upper_cols = columns[::2], columns[1::2]
                masks = [(self.df['Close'] >= self.df[lower]) & (self.df['Close'] <= self.df[upper]) for lower, upper in zip(lower_cols, upper_cols)]
                combined_mask = pd.concat(masks, axis=1).any(axis=1)
    
                if type == 'Exterior':
                    combined_mask = ~combined_mask
    
                self.df[col_name] = combined_mask if requirement else np.where(combined_mask, action_value, 0)
            else:
                # Placeholder for dynamic threshold logic if needed
                pass
        else:
            if method == 'bounds':
                condition = (self.df[metric] >= bounds[0]) & (self.df[metric] <= bounds[1]) if type == 'Interior' else \
                            (self.df[metric] <= bounds[0]) | (self.df[metric] >= bounds[1])
            else:
                condition = (self.df[metric] > threshold) if type == 'Above' else (self.df[metric] < threshold)
            
            self.df[col_name] = condition if requirement else np.where(condition, action_value, 0)

            
    def requirement_cross(self, requirement, wait=2, window=10):
        
        """
        Analyzes a specified column in the DataFrame to detect when a False value changes to True and if this True
        state is sustained for a specified number of periods. It then sets a signal (True) for a defined window 
        of periods following this sustained True condition.
    
        Parameters:
            requirement (str): The name of the column in the DataFrame to analyze. This column should contain boolean values.
            wait (int, optional): The number of consecutive periods the value must remain True after initially changing 
                                  from False to True to consider the change sustained. Defaults to 2.
            window (int, optional): The number of periods for which to set the signal to True following the sustained True condition.
                                    Defaults to 10.
    
        Returns:
            None: This method modifies the DataFrame in-place by adding a new column named 'cross_signal' to indicate where the
                  signal is True based on the defined criteria.
    
        Detailed Explanation:
        - **Cross Detection**: The method first identifies points where the value in the 'requirement' column changes 
                               from False to True.
        - **Sustained True**: It then checks if this True state is sustained for at least 'wait' consecutive periods.
        - **Signal Creation**: If the True state is sustained, the method marks the 'cross_signal' column as True for 
                               'window' periods starting from the sustained True condition.
    
        Example Usage:
            Assuming an instance `data` with a DataFrame that includes a boolean column 'condition_met', you might call:
            data.requirement_cross(requirement='condition_met', wait=3, window=15)
            This sets 'cross_signal' to True for 15 periods following any point where 'condition_met' stays True for 
            at least 3 consecutive periods after initially changing from False to True.
    
        Notes:
        - This method assumes that the DataFrame and the 'requirement' column already exist.
        - It is useful for identifying and signaling prolonged conditions in time series or event-based data analysis.
        """
        
        # Get the column that indicates the requirement
        column = self.df[requirement]
        
        # Identify where the requirement changes from False to True
        cross_from_false_to_true = (column.shift(1) == False) & (column == True)
        
        # Create a series to mark where the requirement has remained True for 'wait' periods
        sustained_true = column.rolling(window=wait, min_periods=wait).sum() == wait
        
        # Combine the crossover and sustained True conditions
        cross_signal = cross_from_false_to_true & sustained_true.shift(-(wait-1))
        
        # Initialize the new column with False values
        self.df['cross_signal'] = False
        
        # Create a Series of ones with the same index as the DataFrame
        ones = pd.Series(1, index=self.df.index)
    
        # Create a cumulative sum that resets at each True in cross_signal
        cumsum = (ones * ~cross_signal).cumsum()
        
        # Subtract the cumsum from itself shifted by window, and check if it's less than window
        window_mask = cumsum - cumsum.shift(window, fill_value=cumsum.iloc[0]) < window
        
        # Set cross_signal to True where the window_mask is True
        self.df['cross_signal'] = window_mask

        
        
    def show_metrics(self):
        print('Available metrics are: {}'.format(self.metrics))
            
        
    def get_signals(self, independent, meet_requirements, markers_to_use):
        
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
            buy_condition = (self.df[markers_to_use] == 1).all(axis=1) & self.df['confluence']
            sell_condition = (self.df[markers_to_use] == -1).all(axis=1) & self.df['confluence']
        else:
            # Any marker can indicate a buy or sell while meeting confluence requirements
            buy_condition = (self.df[markers_to_use] == 1).any(axis=1) & self.df['confluence']
            sell_condition = (self.df[markers_to_use] == -1).any(axis=1) & self.df['confluence']
        
        # Store markers for reference
        self.markers_to_use = markers_to_use
        self.independent = independent
    
        # Create DataFrames for buy and sell signals where conditions are met
        self.buy_df = self.df.loc[buy_condition, ['Close']]
        self.sell_df = self.df.loc[sell_condition, ['Close']]
    
        # Optionally print buy and sell times for review
        print(self.buy_df)
        print('Buy Times')
        print(self.sell_df)
        print('Sell Times')

        
    
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
        group = 'POC'
        for i, poc in enumerate(self.pocs):
            df_data[f'{group}_{i + 1}_POC'] = poc['poc']
            df_data[f'{group}_{i + 1}_Lower'] = poc['xL']
            df_data[f'{group}_{i + 1}_Upper'] = poc['xR']
            df_data[f'{group}_{i + 1}_Prominence'] = poc['prominence']
            df_data[f'{group}_{i + 1}_Density'] = poc['density']
    
        final_index = close.index[-1]
        poc_df = pd.DataFrame(df_data, index=[final_index])
        return poc_df


                
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
            x=self.buy_df.index,
            y=self.buy_df['Close'] * 0.995,  # slightly below the close price for visibility
            mode='markers',
            marker=dict(color='green', symbol='triangle-up'),
            name='Buy Signal'
        ), secondary_y=True)
        fig.add_trace(go.Scatter(
            x=self.sell_df.index,
            y=self.sell_df['Close'] * 1.005,  # slightly above the close price for visibility
            mode='markers',
            marker=dict(color='red', symbol='triangle-down'),
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
        fig.update_yaxes(title="Price $", secondary_y=True, showgrid=True)
        fig.show()

        
        
# Get historical data
data = Data()
data.symbol = 'ETHUSDT'
data.interval = '5m'
data.start_str = '2024-08-23' #YYYY-MM-DD

data.get_historical_klines(show_dataframe = True)


#%%

# Using show = True plots the volume profile at each interval
data.create_bounds(dynamic=True, step=1, window_size=300, name='VPF', func=partial(data.get_volume_profile, bins=50, prom_factor=0.5, show=False))

#%%
#data.add('RSI_14', ta.RSI(data.df['Close'], timeperiod=14))
#data.add_range_marker('RSI_14', type = 'Above', dynamic = False, threshold = 70, action = 'long') 
#data.add_range_marker('RSI_14', type = 'Below', dynamic = False, threshold = 30, action = 'short') 
#data.requirement_cross('POC_Range_Signal_Exterior', wait = 1, window = 5)
#data.add('MA_30', ta.MA(data.df['Close'], timeperiod=30, matype=0))
#data.add('VWAP', (data.df['Volume']*(data.df['High']+data.df['Low'])/2).cumsum() / data.df['Volume'].cumsum())
#data.add('MFI_14', ta.MFI(data.df['Open'], data.df['High'], data.df['Low'], data.df['Close'], timeperiod = 14))
#data.add('CDL3LINESTRIKE', ta.CDL3LINESTRIKE(data.df['Open'], data.df['High'], data.df['Low'], data.df['Close'])/100)
#data.add('CDLBREAKAWAY', ta.CDLBREAKAWAY(data.df['Open'], data.df['High'], data.df['Low'], data.df['Close'])/100)
#data.add('CDLGRAVESTONEDOJI', ta.CDLGRAVESTONEDOJI(data.df['Open'], data.df['High'], data.df['Low'], data.df['Close'])/100)
#data.add_range_marker('MFI_14', type = 'Exterior', bounds = [30,70])

data.get_signals(independent = True, meet_requirements = [], markers_to_use = [])

#%%
data.show(show_volume = True, add_line = False, lines = [], show_pocs = True, use_densities = True)

