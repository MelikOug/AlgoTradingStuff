from binance.client import Client
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import talib as ta
import numpy as np
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
from scipy import stats, signal
import plotly.colors as pc
import matplotlib.cm as cm
import matplotlib.colors as mcolors
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
        
        
    def get_historical_klines(self, show_dataframe = False):
        """
        Get historical kline (candlestick) data from Binance.
    
        :param symbol: The symbol to fetch data for (e.g., 'BTCUSDT').
        :param interval: The interval for kline data (e.g., '1d' for daily data).
        :param start_str: The start date string in format 'YYYY-MM-DD'.
        :param end_str: The end date string in format 'YYYY-MM-DD' (optional).
        :return: A pandas DataFrame containing the kline data.
        """
        # Fetch kline data from Binance
        klines = self.client.get_historical_klines(self.symbol, self.interval, self.start_str, self.end_str)
    
        # Create a DataFrame
        self.df = pd.DataFrame(klines, columns=[
            'Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'close_time',
            'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume', 'ignore'
        ])
    
        # Convert timestamp to datetime
        self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'], unit='ms')
    
        # Set timestamp as index
        self.df.set_index('Timestamp', inplace=True)
    
        # Drop unnecessary columns
        self.df.drop(['close_time', 'quote_asset_volume', 'number_of_trades', 
                 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 
                 'ignore'], axis=1, inplace=True)
    
        self.df = self.df.astype(float) # Convert data from string to float
        
        if show_dataframe == True:
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
        
        
    def create_bounds(self, func, name, step, window_size, dynamic = False, **func_kwargs):
        if dynamic == False:
            # Call func to get a single row of data at the starting index (or another relevant index)
            row = func(window_size=window_size, **func_kwargs)
            
            # Assign each value from the row to a new column, leaving NaNs elsewhere
            for column in row.columns:
                self.df[f'{name}_{column}'] = row[column].iloc[0]
    
            # Use fillna method to propagate the first value down the entire column
            self.df.fillna(method='ffill', inplace=True)
        
        elif dynamic == True:
            # Create an empty DataFrame to hold dynamic bounds
            for i in range(0, len(self.df) - window_size, step):
                #print(i)
                # Call func with specific arguments and the current index i
                row = func(window_size=window_size, index=i, **func_kwargs)
                if i == 0:
                    bounds_df = pd.DataFrame(index = self.df.index, columns = row.columns)                    
                # Add returned row to the bounds DataFrame
                bounds_df.loc[row.index[0]] = row.iloc[0]

        # Merge final dataframe with your original DataFrame
        self.df = pd.concat([self.df, bounds_df], axis=1)
        self.metrics.append(name)
        print(self.metrics)
        
        
    def add_range_marker(self, metric, type = 'Interior', action = 'long', dynamic = False, requirement = False, threshold = 50, bounds = [0,1]):
        '''
        # type: Interior, Exterior, Above, Below
        # action: long, short
        # dynamic only compares close price with moving indicator
        
        if type == 'Interior' or type == 'Exterior':
            method = 'bounds'
        else:
            method = 'threshold'
            
        if action == 'long':
            action = 1
        elif action == 'short':
            action = -1
            
        if metric in self.metrics:
            
            if dynamic == True:
                if method == 'bounds':
                    columns = [col for col in self.df.columns if col.startswith(metric+'_') and ('Lower' in col or 'Upper' in col)]
                    lower_cols = columns[::2]
                    upper_cols = columns[1::2]
                    
                    # Create a boolean mask for each pair of bounds
                    # If close within any of these dynamic bounds, return True
                    masks = [(self.df['Close'] >= self.df[lower]) & (self.df['Close'] <= self.df[upper]) 
                             for lower, upper in zip(lower_cols, upper_cols)]
                    
                    # Combine all masks with OR operation
                    combined_mask = pd.concat(masks, axis=1).any(axis=1)
                    
                    # Invert True and False
                    if type == 'Exterior':
                        combined_mask = ~combined_mask
                    
                    
                    # Assign values based on the combined mask
                    if requirement == True:
                        # Just have indicator show True/False
                        self.df[metric+'_Range_Signal_'+type] = combined_mask
                        # Have indicator show action to take
                    else:
                        self.df[metric+'_Range_Signal_'+type] = np.where(combined_mask, action, 0) # 1/-1 when True, 0 when False
                        
                elif method == 'threshold':

                    if requirement == True:
                        if type == 'Above':    
                            self.df[metric+'_Thresh_Signal_'+type] = self.df[metric].apply(lambda x: True if self.df['Close'] <= x else False)
                        elif type == 'Below':
                            self.df[metric+'_Thresh_Signal_'+type] = self.df[metric].apply(lambda x: True if self.df['Close'] >= x else False)
                    
                    elif requirement == False:
                        if type == 'Above':    
                            self.df[metric+'_Thresh_Signal_'+type] = self.df[metric].apply(lambda x: action if self.df['Close'] <= x else 0)
                        elif type == 'Below':
                            self.df[metric+'_Thresh_Signal_'+type] = self.df[metric].apply(lambda x: action if self.df['Close'] >= x else 0)
                    
                
            elif dynamic == False:
                
                if method == 'bounds':
                    
                    if requirement == True:
                        if type == 'Interior':
                            self.df[metric+'_Range_Signal_'+type] = self.df[metric].apply(lambda x: True if bounds[0] <= x and bounds[1] >= x else False)
                        elif type == 'Exterior':       
                            self.df[metric+'_Range_Signal_'+type] = self.df[metric].apply(lambda x: True if bounds[0] >= x and bounds[1] <= x else False)    
                    
                    elif requirement == False:
                        if type == 'Interior':
                            self.df[metric+'_Range_Signal_'+type] = self.df[metric].apply(lambda x: action if bounds[0] <= x and bounds[1] >= x else 0)
                        elif type == 'Exterior':       
                            self.df[metric+'_Range_Signal_'+type] = self.df[metric].apply(lambda x: action if bounds[0] >= x and bounds[1] <= x else 0)  
                
                elif method == 'threshold':
                    
                    if requirement == True:
                        if type == 'Above':    
                            self.df[metric+'_Thresh_Signal_'+type] = self.df[metric].apply(lambda x: True if threshold <= x else False)
                        elif type == 'Below':
                            self.df[metric+'_Thresh_Signal_'+type] = self.df[metric].apply(lambda x: True if threshold >= x else False)
                    
                    elif requirement == False:
                        if type == 'Above':    
                            self.df[metric+'_Thresh_Signal_'+type] = self.df[metric].apply(lambda x: action if threshold <= x else 0)
                        elif type == 'Below':
                            self.df[metric+'_Thresh_Signal_'+type] = self.df[metric].apply(lambda x: action if threshold >= x else 0)
                    
                        
        else:
            print('{} is not in metrics. The available metrics are: {}'.format(metric, self.metrics))
        
        '''
        
        """
        Adds a range marker based on specified conditions to help identify trading signals within the DataFrame.
    
        Parameters:
            metric (str): The metric to apply the marker to.
            type (str): The type of marker ('Interior', 'Exterior', 'Above', 'Below').
            action (str): The trading action associated with the marker ('long' or 'short').
            dynamic (bool): Whether the marker should consider dynamic boundaries.
            requirement (bool): Whether the marker should indicate presence (True/False) or action.
            threshold (float): The threshold value for 'Above' or 'Below' conditions.
            bounds (list): The lower and upper bounds for 'Interior' or 'Exterior' conditions.
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
            
            
    def requirement_cross(self,requirement, wait = 2, window = 10):
        # Checks requirement column to see if it's changed from False to True
        # Sees if after this cross it has remained True for at least (wait amount of periods)
        # Sets adjacent cross_signal row to True for (window periods)
        
        # Get the column that indicates the requirement
        column = self.df[requirement]
        
        # Identify where the requirement crosses from False to True
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
        # If meet_requirements is not provided or is empty, 
        # set the confluence column in self.df to all True values.
        if not meet_requirements:
            self.df['confluence'] = True
        else:
            # If meet_requirements is provided, set the confluence column in self.df
            # to True only where all columns specified in meet_requirements are True.
            self.df['confluence'] = self.df[meet_requirements].all(axis=1)
        
        # If independent is False, we require all markers_to_use to be 1 for buy signals
        # and -1 for sell signals, and the confluence column must also be True.
        if independent == False:
            buy_condition = (self.df[markers_to_use] == 1).all(axis=1) & self.df['confluence']
            sell_condition = (self.df[markers_to_use] == -1).all(axis=1) & self.df['confluence']
        # If independent is True, we require any marker in markers_to_use to be 1 for buy signals
        # and -1 for sell signals, and the confluence column must also be True.
        elif independent == True:
            buy_condition = (self.df[markers_to_use] == 1).any(axis=1) & self.df['confluence']
            sell_condition = (self.df[markers_to_use] == -1).any(axis=1) & self.df['confluence']
        
        self.markers_to_use = markers_to_use
        self.independent = independent
        self.buy_df = self.df.loc[buy_condition, ['Close']]
        self.sell_df = self.df.loc[sell_condition, ['Close']]

        
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
        close = self.df['Close'][index:index + window_size]
        volume = self.df['Volume'][index:index + window_size]
        kde_factor = 0.05
        num_samples = 500
        kde = stats.gaussian_kde(close, weights=volume, bw_method=kde_factor)
        xr = np.linspace(close.min(), close.max(), num_samples)
        kdy = kde(xr)
        ticks_per_sample = (xr.max() - xr.min()) / num_samples
    
        # Find peaks
        min_prom = kdy.max() * prom_factor
        peaks, peak_props = signal.find_peaks(kdy, prominence=min_prom, width=1)
        pkx = xr[peaks]
        pky = kdy[peaks]
    
        # Widths
        left_ips = peak_props['left_ips']
        right_ips = peak_props['right_ips']
        width_x0 = xr.min() + (left_ips * ticks_per_sample)
        width_x1 = xr.min() + (right_ips * ticks_per_sample)
        width_y = peak_props['width_heights']
    
        if show:
            plt.figure(figsize=(10, 5), dpi = 50)
            plt.grid()
            plt.plot(xr, kdy, label='KDE of Volume Profile')
            # Plot peaks
            plt.plot(pkx, pky, "x", label='Peaks')
    
            # Plot red lines showing widths
            for i in range(len(width_x0)):
                plt.hlines(width_y[i], width_x0[i], width_x1[i], color='red', label='Width' if i == 0 else "")
            print(close.index[-1])
            plt.title('Volume Profile with Peaks and Widths - '+str(close.index[-1]))
            plt.xlabel('Price')
            plt.ylabel('Density')
            plt.legend()
            plt.show()
            plt.close()
        self.pocs = []
    
        # Store POCs data
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

                
    def show(self, show_volume=False, points=0, add_line = True, lines = ['MA_30'], show_pocs = True, use_densities = True):
        trunc_df = self.df.iloc[-points:, :]
        
        # Create the price movement candlesticks
        candlesticks = go.Candlestick(
            x=trunc_df.index,
            open=trunc_df['Open'],
            high=trunc_df['High'],
            low=trunc_df['Low'],
            close=trunc_df['Close'],
            showlegend=False
        )

        # Create the figure with subplots
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add candlestick trace to the figure
        fig.add_trace(candlesticks, secondary_y=True)

        # Add trace for POCs (Only works if get_volume_profile is called)
        if show_pocs == True:
            
            pocs = self.df.filter(like='_POC', axis=1)
            uppers = self.df.filter(like='_Upper', axis=1)
            lowers = self.df.filter(like='_Lower', axis=1)
            proms = self.df.filter(like='_Prominence', axis=1)
            ds = self.df.filter(like='_Density', axis=1)
            
            # Group data into the desired format
            grouped_data = [
                [pocs.iloc[:, i], uppers.iloc[:, i], lowers.iloc[:, i], proms.iloc[:, i], ds.iloc[:, i]]
                for i in range(pocs.shape[1])
            ]
            
            for group in grouped_data:
                # Add POC line                
                fig.add_trace(go.Scatter(
                x=group[0].index,
                y=group[0].values,
                mode='markers',
                marker=dict(color='Gray', size=1),
                name=group[0].name
                ), secondary_y=True)
                
                # Add Upper line                
                fig.add_trace(go.Scatter(
                x=group[1].index,
                y=group[1].values,
                mode='markers',
                marker=dict(color='Green', size=1),
                name=group[1].name
                ), secondary_y=True)
                
                # Add Lower line                
                fig.add_trace(go.Scatter(
                x=group[2].index,
                y=group[2].values,
                mode='markers',
                marker=dict(color='Red', size=1),
                name=group[2].name
                ), secondary_y=True)
    
                
        # Show volume bars underneath and use a different y-axis
        if show_volume == True:
            volume_bars = go.Bar(
                x=trunc_df.index,
                y=trunc_df['Volume'],
                showlegend=False,
                marker={
                    "color": "rgba(128,128,128,7)",
                }
            )
            fig.add_trace(volume_bars, secondary_y=False)
            fig.update_yaxes(title="Volume", secondary_y=False, showgrid=False)
        

        # Add trace for Buy signals
        fig.add_trace(go.Scatter(
            x=self.buy_df.index,
            y=self.buy_df['Close']*0.995,
            mode='markers',
            marker=dict(color='green', symbol = 'triangle-up'),
            name='Buy Signal'
        ), secondary_y=True)
    
        
        # Add trace for Sell signals
        fig.add_trace(go.Scatter(
            x=self.sell_df.index,
            y=self.sell_df['Close']*1.005,
            mode='markers',
            marker=dict(color='red', symbol = 'triangle-down'),
            name='Sell Signal'
        ), secondary_y=True)
        
        

        # Add trace for additional TA line(s)
        if add_line == True: 
            for line in lines:
                if line in self.metrics:
                    fig.add_trace(go.Scatter(
                        x=self.df.index,
                        y=self.df[line],
                        name = line,
                        showlegend = True,
                    ), secondary_y=True)
                    
        
        # Format figure
        fig.update_layout(
            title='{} - {}, Markers: {}, Independent = {}'.format(self.symbol, self.interval, self.markers_to_use, self.independent),
            height=800,
            # Hid/Show Plotly scrolling minimap below the price chart
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
