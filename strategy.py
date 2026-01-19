from datetime import timedelta
import pandas as pd
import numpy as np
import talib

import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt


class Ichimoku(object):

    def __init__(self):
        pass

    def indicators(self, df):
        h_9 = df['h'].rolling(window=9).max()
        l_9 = df['l'].rolling(window=9).min()

        df['tenkan_sen'] = (h_9 + l_9) / 2

        high_26 = df['h'].rolling(window=26).max()
        low_26 = df['l'].rolling(window=26).min()
        df['kijun_sen'] = (high_26 + low_26) / 2

        last_index = df.iloc[-1:].index[0]
        last_date = df['date'].iloc[-1].date()
        for i in range(26):
            df.loc[last_index + 1 + i, 'date'] = last_date + timedelta(days=i)

        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)

        high_52 = df['h'].rolling(window=52).max()
        low_52 = df['l'].rolling(window=52).min()
        df['senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)

        # most charting softwares dont plot this line
        df['chikou_span'] = df['mid'].shift(-22)  # sometimes -26

        copy = df.copy()
        # copy = copy.set_index('date')
        tmp = copy[['mid', 'senkou_span_a', 'senkou_span_b', 'kijun_sen', 'tenkan_sen']].tail(300)
        a1 = tmp.plot()
        a1.fill_between(tmp.index, tmp.senkou_span_a, tmp.senkou_span_b, alpha=0.5)
        plt.show()

        return df


class Volume(object):
    def __init__(self):
        pass

    @staticmethod
    def indicators(df):
        df['obv'] = talib.OBV(df['mid'], df['volume'])
        df['ad'] = talib.AD(df['mid_h'], df['mid_l'], df['mid'], df['volume'])
        df['rsi'] = talib.RSI(df['mid'], timeperiod=8)

        copy = df.copy()
        copy = copy.set_index('date')
        fig = plt.figure()

        ax1 = fig.add_subplot(311, ylabel='Price')
        copy[['mid']].plot(ax=ax1)

        ax1.grid()

        ax2 = fig.add_subplot(312, ylabel='Volume')
        copy[['volume']].plot(ax=ax2)
        ax2.grid()

        ax3 = fig.add_subplot(313, ylabel='RSI')
        copy[['rsi']].plot(ax=ax3)
        ax3.fill_between(copy.index, y1=25, y2=75, color='#adccff', alpha='0.3')
        ax3.grid()

        plt.show()


class Scalping(object):
    def __init__(self, market, units, window=9):
        self.market = market
        self.units = units
        self.window = window

    def indicators(self, df):
        ema_50 = df['ask'].rolling(window=50).mean()
        df['ema_50'] = ema_50
        ema_100 = df['ask'].rolling(window=100).mean()
        df['ema_100'] = ema_100
        df['rsi'] = talib.RSI(df['ask'], timeperiod=self.window)

        df['buy'] = np.where((df['ema_50'] > df['ema_100']) & (df['rsi'] < 30), 1.0, 0.0)
        df['sell'] = np.where((df['ema_50'] < df['ema_100']) & (df['rsi'] > 70), 1.0, 0.0)

        portfolio = backtest(df=df, units=self.units, market=self.market)

        copy = df.copy()
        copy = copy.set_index('date')
        fig = plt.figure()
        ax1 = fig.add_subplot(311, ylabel='Price')
        copy[['ask', 'ema_50', 'ema_100']].plot(ax=ax1)
        ax1.scatter(x=portfolio.index, y=portfolio.loc[portfolio.index, 'price'], color='red')
        ax1.grid()

        # ax3 = fig.add_subplot(312, ylabel='Positions')
        # copy[['buy', 'sell']].plot(ax=ax3)

        ax2 = fig.add_subplot(313, ylabel='RSI')
        copy[['rsi']].plot(ax=ax2)
        ax2.fill_between(copy.index, y1=30, y2=70, color='#adccff', alpha='0.3')

        # ax4 = fig.add_subplot(224, ylabel='MACD')
        # copy[['macd', 'macdsignal']].plot(ax=ax4)

        plt.show()


class BollinguerRsiMacd(object):

    def __init__(self, market, units, window):
        self.market = market
        self.units = units
        self.window = window
        print(self.window)
        print(self.market)

    def indicators(self, df, rsi_upper=70, rsi_lower=30, timeperiod=9):
        # df['ema']=df['ask'].rolling(window=200, min_periods=1).mean()
        df['ema'] = talib.EMA(df['mid'], timeperiod=200)
        df['ema_9'] = talib.EMA(df['mid'], timeperiod=9)
        df['ema_14'] = talib.EMA(df['mid'], timeperiod=14)

        df['long'] = df['mid'].rolling(window=self.window, min_periods=1).mean()
        # df['long'] = talib.EMA(df['ask'], timeperiod= self.window)
        rolling_std = df['mid'].rolling(window=self.window, min_periods=1).std()
        # Bollinguer bands
        df['upper'] = df['long'] + (rolling_std * 2)
        df['lower'] = df['long'] - (rolling_std * 2)

        # up, mid, low = talib.BBANDS(df['ask'], timeperiod=self.window, nbdevup=2, nbdevdn=2, matype=0)

        # print((up==low))

        # df['upper'] = up
        # df['lower'] = low
        df['bbp'] = (df['ask'] - df['lower']) / (df['upper'] - df['lower'])
        # df['upper'], df['middle'], df['lower'] = \
        #   talib.BBANDS(df['ask'], timeperiod=self.window, nbdevup=2, nbdevdn=2, matype=0)
        df['rsi'] = talib.RSI(df['ask'], timeperiod=timeperiod)
        df['macd'], df['macdsignal'], df['macdhist'] = talib.MACD(df['ask'], fastperiod=12, slowperiod=26,
                                                                  signalperiod=9)
        # estrategia: Cuando el precio de compra sea igual o mayor que el limite inferior: Compra
        # df['buy'] = np.where(df['mid'] <= df['lower'], 1.0, 0.0)
        # Cuando el precio de venta sea mayor o igual que el limite superior: Venta
        # df['sell'] = np.where(df['mid'] >= df['upper'], 1.0, 0.0)

        # estrategia:
        df['buy'] = np.where((df['rsi'] < rsi_lower) & (df['bbp'] < 0), 1.0, 0.0)
        df['sell'] = np.where((df['rsi'] > rsi_upper) & (df['bbp'] > 1), 1.0, 0.0)

        # print(df[['date', 'buy', 'sell', 'rsi', 'bbp', 'macd', 'macdsignal']].tail())

        portfolio = backtest(df=df, units=self.units, market=self.market)

        copy = df.copy()
        copy = copy.set_index('date')
        fig = plt.figure()
        ax1 = fig.add_subplot(211, ylabel='Price')
        copy[['ask', 'upper', 'lower']].plot(ax=ax1)
        ax1.scatter(x=portfolio.index, y=portfolio.loc[portfolio.index, 'price'], color='red')
        ax1.grid()

        # ax3 = fig.add_subplot(312, ylabel='Positions')
        # copy[['buy', 'sell']].plot(ax=ax3)

        ax2 = fig.add_subplot(212, ylabel='RSI')
        copy[['rsi']].plot(ax=ax2)
        ax2.fill_between(copy.index, y1=rsi_lower, y2=rsi_upper, color='#adccff', alpha=0.3)

        # ax4 = fig.add_subplot(224, ylabel='MACD')
        # copy[['macd', 'macdsignal']].plot(ax=ax4)

        plt.show()

        return df.shift(1)

    def backtest(self, df):
        open_position = False
        bought = []
        sold = []
        signal = []
        date_index = []
        for i in df.index:
            if self.market == 'bull':
                print("Entramos en bull!!")
                if df.loc[i, 'buy'] == 1 and open_position is False:
                    open_position = True
                    print('Buying signal. %f Date: %s' % (df.loc[i, 'ask'] * self.units, df.loc[i, 'date']))
                    bought.append(df.loc[i, 'ask'] * self.units)
                    sold.append(0.0)
                    signal.append(df.loc[i, 'ask'])
                    date_index.append(df.loc[i, 'date'])
                elif df.loc[i, 'sell'] == 1 and open_position is True:
                    open_position = False
                    print('Selling signal. %f Date: %s' % (df.loc[i, 'bid'] * self.units, df.loc[i, 'date']))
                    sold.append(df.loc[i, 'bid'] * self.units)
                    bought.append(0.0)
                    signal.append(df.loc[i, 'bid'])
                    date_index.append(df.loc[i, 'date'])
            else:
                if df.loc[i, 'sell'] == 1 and open_position is False:
                    open_position = True
                    print('Selling signal. %f Date: %s' % (df.loc[i, 'bid'] * self.units, df.loc[i, 'date']))
                    sold.append(df.loc[i, 'bid'] * self.units)
                    bought.append(0.0)
                    signal.append(df.loc[i, 'bid'])
                    date_index.append(df.loc[i, 'date'])
                elif df.loc[i, 'buy'] == 1 and open_position is True:
                    open_position = False
                    print('Buying signal. %f Date: %s' % (df.loc[i, 'ask'] * self.units, df.loc[i, 'date']))
                    bought.append(df.loc[i, 'ask'] * self.units)
                    sold.append(0.0)
                    signal.append(df.loc[i, 'ask'])
                    date_index.append(df.loc[i, 'date'])

        portfolio = pd.DataFrame(data={'bought': bought, 'sold': sold, 'price': signal},
                                 columns=['bought', 'sold', 'price'],
                                 index=date_index).fillna(0.0)

        portfolio['return'] = portfolio['sold'] + portfolio['bought']
        portfolio['MR'] = np.log(portfolio['price'] / portfolio['price'].shift(1))
        portfolio['pl'] = portfolio['MR'] * self.units
        # portfolio['pct_change']= portfolio['price'].pct_change(1).shift(1) * self.units
        portfolio['pct_change'] = portfolio['pl'].pct_change(1)

        print(portfolio[['price', 'bought', 'sold']].tail())

        print('Sumatorio %s P&L %s' % (portfolio['sold'].sum() - portfolio['bought'].sum(), portfolio['pl'].sum()))

        # portfolio['pct_change'].cumsum().plot()
        # plt.show()

        return portfolio


class BbandsAdxStrategy(object):
    def __init__(self, market, units, windows):
        self.market = market
        self.units = units
        self.window = windows

    def indicators(self, df, rsi_upper=70, rsi_lower=30, time_period=20):
        df['rsi'] = talib.RSI(df['mid'], timeperiod=self.window)
        df['adx'] = talib.ADX(df['mid_h'], df['mid_l'], df['mid'], timeperiod=self.window)
        df['upper'], df['middle'], df['lower'] = talib.BBANDS(df['mid'], timeperiod=time_period, nbdevup=2, nbdevdn=2)
        df['bbp'] = (df['mid'] - df['lower']) / (df['upper'] - df['lower'])

        df['buy'] = np.where((df['bbp'] < 0) & (df['adx'] < 40), 1.0, 0.0)
        df['sell'] = np.where((df['bbp'] > 1) & (df['adx'] < 40), 1.0, 0.0)

        portfolio = backtest(df=df, market=self.market, units=self.units)

        copy = df.copy()
        copy = copy.set_index('date')
        fig = plt.figure()
        ax1 = fig.add_subplot(313, ylabel='BBANDS')
        copy[['upper', 'middle', 'lower']].plot(ax=ax1)
        # ax1.fill_between(copy.index, y1=rsi_lower, y2=rsi_upper, color='#adccff', alpha='0.3')
        ax1.grid()

        ax3 = fig.add_subplot(312, ylabel='ADX')
        copy[['adx']].plot(ax=ax3)
        ax3.fill_between(copy.index, y1=40, y2=40, color='#adccff', alpha='0.3')
        ax3.grid()
        # ax4 = fig.add_subplot(312, ylabel='Positions')
        # copy[['buy', 'sell']].plot(ax=ax4)

        ax5 = fig.add_subplot(311, ylabel='Price')
        copy[['mid', 'upper', 'lower']].plot(ax=ax5)
        ax5.scatter(x=portfolio.index, y=portfolio.loc[portfolio.index, 'price'], color='red')
        ax5.grid()

        plt.show()

    def backtest(self, df):
        pass


class AdxRsiStrategy(object):
    def __init__(self, market, units, windows):
        self.market = market
        self.units = units
        self.window = windows

    def indicators(self, df, upper=70, lower=30, time_period=14):
        df['rsi'] = talib.RSI(df['mid'], timeperiod=self.window)
        df['adx'] = talib.ADX(df['mid_h'], df['mid_l'], df['mid'], timeperiod=self.window)

        _adx = 50
        if self.market == 'bull':
            # bullish
            df['buy'] = np.where((df['rsi'] > upper) & (df['adx'] > _adx), 1.0, 0.0)
            df['sell'] = np.where((df['rsi'] < upper), 1.0, 0.0)
        else:
            # Bearish
            df['buy'] = np.where((df['rsi'] > lower), 1.0, 0.0)
            df['sell'] = np.where((df['rsi'] < lower) & (df['adx'] > _adx), 1.0, 0.0)

        portfolio = backtest(df=df, market=self.market, units=self.units)

        copy = df.copy()
        copy = copy.set_index('date')
        fig = plt.figure()
        ax1 = fig.add_subplot(313, ylabel='RSI')
        copy[['rsi']].plot(ax=ax1)
        ax1.fill_between(copy.index, y1=lower, y2=upper, color='#adccff', alpha='0.3')
        ax1.grid()

        ax3 = fig.add_subplot(312, ylabel='ADX')
        copy[['adx']].plot(ax=ax3)
        ax3.grid()
        # ax4 = fig.add_subplot(312, ylabel='Positions')
        # copy[['buy', 'sell']].plot(ax=ax4)

        ax5 = fig.add_subplot(311, ylabel='Price')
        copy[['mid']].plot(ax=ax5)
        ax5.scatter(x=portfolio.index, y=portfolio.loc[portfolio.index, 'price'], color='red')
        ax5.grid()

        plt.show()


class RsiFlagStrategy(object):
    def __init__(self, market, units, windows):
        self.market = market
        self.units = units
        self.window = windows

    def indicators(self, df, upper=70, lower=30):
        # df['ema'] = df['ask'].rolling(window=200, min_periods=1).mean()

        df['rsi'] = talib.RSI(df['mid'], timeperiod=self.window)
        df['willr'] = talib.WILLR(df['mid_h'], df['mid_l'], df['mid'], timeperiod=self.window)

        df['low'] = np.where((df['rsi'] < lower) & (df['willr'] < -80), 1.0, 0.0)
        df['upper'] = np.where((df['rsi'] > upper) & (df['willr'] > -20), 1.0, 0.0)

        df['buy'] = np.where((df['low'] - df['low'].shift(1) == -1), 1.0, 0.0)
        df['sell'] = np.where((df['upper'] - df['upper'].shift(1) == -1), 1.0, 0.0)

        # print(df[['rsi', 'low', 'upper', 'buy', 'sell']])

        portfolio = backtest(df=df, market=self.market, units=self.units)

        copy = df.copy()
        copy = copy.set_index('date')
        fig = plt.figure()
        ax1 = fig.add_subplot(313, ylabel='RSI')
        copy[['rsi']].plot(ax=ax1)
        ax1.fill_between(copy.index, y1=lower, y2=upper, color='#adccff', alpha=0.3)
        ax1.grid()

        ax3 = fig.add_subplot(312, ylabel='William %R')
        copy[['willr']].plot(ax=ax3)
        ax3.fill_between(copy.index, y1=-80, y2=-20, color='#adccff', alpha=0.3)
        ax3.grid()

        ax5 = fig.add_subplot(311, ylabel='Price')
        copy[['mid']].plot(ax=ax5)
        ax5.scatter(x=portfolio.index, y=portfolio.loc[portfolio.index, 'price'],
                    color=portfolio.loc[portfolio.index, 'color'])

        ax5.grid()

        plt.show()


class RsiMomentumStrategy(object):
    def __init__(self, market, units, windows):
        self.market = market
        self.units = units
        self.window = windows

    def indicators(self, df, upper=70, lower=30):
        # df['ema'] = df['ask'].rolling(window=200, min_periods=1).mean()
        df['ema'] = talib.SMA(df['mid'], timeperiod=150)
        df['rsi'] = talib.RSI(df['mid'], timeperiod=self.window)
        df['mom'] = talib.MOM(df['mid'], timeperiod=self.window)
        df['adx'] = talib.ADX(df['mid_h'], df['mid_l'], df['mid'], timeperiod=self.window)
        df['willr'] = talib.WILLR(df['mid_h'], df['mid_l'], df['mid'], timeperiod=self.window)

        df['buy'] = np.where((df['rsi'] < lower) & (df['mom'] < 0) & (df['willr'] < -80), 1.0, 0.0)
        df['sell'] = np.where((df['rsi'] > upper) & (df['mom'] > 0) & (df['willr'] > -20), 1.0, 0.0)

        df['trend'] = np.where(df['mid'] > df['ema'], 0, 1)

        # print(df[['rsi', 'mom', 'ema', 'mid', 'buy', 'sell', 'trend']])

        portfolio = backtest(df=df, market=self.market, units=self.units)

        copy = df.copy()
        copy = copy.set_index('date')
        fig = plt.figure()
        ax1 = fig.add_subplot(313, ylabel='RSI')
        copy[['rsi']].plot(ax=ax1)
        ax1.fill_between(copy.index, y1=lower, y2=upper, color='#adccff', alpha=0.3)
        ax1.grid()

        ax3 = fig.add_subplot(312, ylabel='Will %R')
        copy[['willr']].plot(ax=ax3)
        ax3.fill_between(copy.index, y1=-20, y2=-80, color='#adccff', alpha=0.3)
        ax3.grid()
        # ax4 = fig.add_subplot(312, ylabel='Positions')
        # copy[['buy', 'sell']].plot(ax=ax4)

        ax5 = fig.add_subplot(311, ylabel='Price')
        copy[['mid', 'ema']].plot(ax=ax5)
        ax5.scatter(x=portfolio.index, y=portfolio.loc[portfolio.index, 'price'],
                    color=portfolio.loc[portfolio.index, 'color'])
        ax5.grid()

        plt.show()

    def backtest(self, df):
        open_position = False
        bought = []
        sold = []
        date_index = []
        for i in df.index:
            # bull
            if self.market == 'bull':
                if df.loc[i, 'buy'] == 1 and open_position is False:
                    open_position = True
                    print('Buying signal. %f Date: %s' % (df.loc[i, 'ask'] * self.units, df.loc[i, 'date']))
                    bought.append(df.loc[i, 'ask'] * self.units)
                    sold.append(0.0)
                    date_index.append(df.loc[i, 'date'])
                elif df.loc[i, 'sell'] == 1 and open_position is True:
                    open_position = False
                    print('Selling signal. %f Date: %s' % (df.loc[i, 'bid'] * self.units, df.loc[i, 'date']))
                    sold.append(df.loc[i, 'bid'] * self.units)
                    bought.append(0.0)
                    date_index.append(df.loc[i, 'date'])
            else:
                if df.loc[i, 'sell'] == 1 and open_position is False:
                    open_position = True
                    print('Selling signal. %f Date: %s' % (df.loc[i, 'ask'] * self.units, df.loc[i, 'date']))
                    sold.append(df.loc[i, 'ask'] * self.units)
                    bought.append(0.0)
                    date_index.append(df.loc[i, 'date'])
                elif df.loc[i, 'buy'] == 1 and open_position is True:
                    open_position = False
                    print('Buying signal. %f Date: %s' % (df.loc[i, 'bid'] * self.units, df.loc[i, 'date']))
                    bought.append(df.loc[i, 'bid'] * self.units)
                    sold.append(0.0)
                    date_index.append(df.loc[i, 'date'])

        portfolio = pd.DataFrame(data={'bought': bought, 'sold': sold}, columns=['bought', 'sold'],
                                 index=date_index).fillna(0.0)
        portfolio['return'] = portfolio['sold'] + portfolio['bought']
        portfolio['pct_change'] = portfolio['return'].pct_change()

        print(portfolio)

        print('Sumatorio %s' % (portfolio['sold'].sum() - portfolio['bought'].sum()))

        # portfolio['pct_change'].cumsum().plot()
        # plt.show()

        return portfolio


class MACDRSIStrategy(object):
    def __init__(self, market, units, windows):
        self.market = market
        self.units = units
        self.window = windows

    def indicators(self, df, upper=70, lower=30):
        df['macd'], df['macdsignal'], df['macdhist'] = talib.MACD(df['mid'], fastperiod=12, slowperiod=26,
                                                                  signalperiod=9)
        df['rsi'] = talib.RSI(df['mid'], timeperiod=self.window)
        df['ema'] = talib.EMA(df['mid'], timeperiod=150)

        df['low'] = np.where((df['rsi'] < lower) , 1.0, 0.0)
        df['upper'] = np.where((df['rsi'] > upper) , 1.0, 0.0)

        df['buy'] = np.where((df['macd'] > df['macdsignal'])
                             & df['macdhist'] > 0, 1.0, 0.0)
        df['sell'] = np.where((df['macd'] < df['macdsignal'])
                              & df['macdhist'] < 0, 1.0, 0.0)

        print(df[['macd', 'macdsignal', 'macdhist', 'buy', 'sell']].loc[50:])

        portfolio = backtest(df=df, market=self.market, units=self.units)
        copy = df.copy()
        copy = copy.set_index('date')

        fig = plt.figure()

        ax1 = fig.add_subplot(222, ylabel='MACD')
        copy[['macd', 'macdsignal', 'macdhist']].plot(ax=ax1)
        #ax1.bar(copy.index, copy['macdhist'], color='green')
        ax1.grid()

        #ax4 = fig.add_subplot(223, ylabel='MACD')
        #copy[['macdhist']].plot(ax=ax4)
        #ax4.bar(copy.index, height=copy['macdhist'], color='orange')
        #ax4.grid()

        ax2 = fig.add_subplot(224, ylabel='RSI')
        copy[['rsi']].plot(ax=ax2)
        ax2.fill_between(copy.index, y1=lower, y2=upper, color='#adccff', alpha=0.3)
        ax2.grid()

        ax3 = fig.add_subplot(221, ylabel='Price')
        copy[['ask']].plot(ax=ax3)
        ax3.scatter(x=portfolio.index, y=portfolio.loc[portfolio.index, 'price'], color='red')
        ax3.grid()

        plt.show()

        return df

    def backtest(self, df):
        pass


def backtest(df, market, units):
    open_position = False
    bought = []
    sold = []
    signal = []
    date_index = []
    _red = []
    _green = []

    for i in df.index:
        if market == 'long':
            if df.loc[i, 'buy'] == 1 and open_position is False:
                open_position = True
                # print('Buying signal. %f Date: %s' % (df.loc[i, 'ask'] * units, df.loc[i, 'date']))
                bought.append(df.loc[i, 'ask'])
                sold.append(0.0)
                signal.append(df.loc[i, 'ask'])
                date_index.append(df.loc[i, 'date'])
            elif df.loc[i, 'sell'] == 1 and open_position is True:
                open_position = False
                # print('Selling signal. %f Date: %s' % (df.loc[i, 'bid'] * units, df.loc[i, 'date']))
                sold.append(df.loc[i, 'bid'])
                bought.append(0.0)
                signal.append(df.loc[i, 'bid'])
                date_index.append(df.loc[i, 'date'])
        else:
            if df.loc[i, 'sell'] == 1 and open_position is False:
                open_position = True
                # print('Selling signal. %f Date: %s' % (df.loc[i, 'bid'] * units, df.loc[i, 'date']))
                sold.append(df.loc[i, 'bid'])

                bought.append(0.0)
                signal.append(df.loc[i, 'bid'])
                date_index.append(df.loc[i, 'date'])
            elif df.loc[i, 'buy'] == 1 and open_position is True:
                open_position = False
                # print('Buying signal. %f Date: %s' % (df.loc[i, 'ask'] * units, df.loc[i, 'date']))
                bought.append(df.loc[i, 'ask'])

                sold.append(0.0)
                signal.append(df.loc[i, 'ask'])
                date_index.append(df.loc[i, 'date'])

    portfolio = pd.DataFrame(data={'bought': bought, 'sold': sold, 'price': signal},
                             columns=['bought', 'sold', 'price'],
                             index=date_index).fillna(0.0)

    portfolio['return'] = np.where(market == 'long', (portfolio['sold'] - portfolio['bought'].shift(1)) * units,
                                   (portfolio['sold'].shift(1) - portfolio['bought'])* units)
    portfolio['color'] = np.where(portfolio['sold'] != 0.0, 'red', 'lime')

    portfolio['MR'] = np.log(portfolio['price'] / portfolio['price'].shift(1))
    portfolio['pl'] = portfolio['MR'] * units
    # portfolio['pct_change']= portfolio['price'].pct_change(1).shift(1) * self.units
    portfolio['pct_change'] = portfolio['return'].pct_change()

    print(portfolio[['bought', 'sold', 'color', 'return']])

    print('Sumatorio %s ' % (portfolio['return'].sum()))

    return portfolio
