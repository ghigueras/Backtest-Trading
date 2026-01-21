import sys
import argparse
sys.path.append('/users/ghigueras/AI/PythonProject/OandaBroker/src')
import logging

from logging.handlers import TimedRotatingFileHandler

from oandapyV20 import API

from Authenticate import Authenticate
from strategy import Ichimoku, BollinguerRsiMacd, RsiMomentumStrategy, MACDRSIStrategy, Volume, \
    BbandsAdxStrategy, AdxRsiStrategy, RsiFlagStrategy
from pricing import Prices


class Logger(object):
    def __init__(self, name):
        logging.basicConfig(level='INFO', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.log = logging.getLogger(name=name)

    def getLoggin(self):
        # create a file handler
        # handler = logging.FileHandler('DailyTrading.log')
        handler = TimedRotatingFileHandler('Backtest.log',
                                           when="d",
                                           interval=1,
                                           backupCount=10)
        handler.setLevel(logging.INFO)

        # create a logging format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # add the handlers to the logger
        self.log.addHandler(handler)

        return self.log


if __name__ == '__main__':

    log = Logger(name=__name__).getLoggin()

    args = argparse.ArgumentParser(description="Environament y tendencia")
    args.add_argument("-e", help="Environment a usar: practice o live", dest="env", required= True)
    args.add_argument("-m", help="Sentido del mercado: long o short", dest="mkt", required= True)
    args.add_argument("-i", help="Instrumento a operar", dest="inst", required= True)
    args.add_argument("-t", help="Timeframe a operar", dest="tmf", required= True)
    argumentos = args.parse_args()

    log.info("Argumentos %s, %s, %s, %s", argumentos.env, argumentos.mkt, argumentos.inst, argumentos.tmf)
    #print(argumentos.mkt)
    
    # valido que el mercado es uno de los esperados
    tendencias = ["long", "short"]
    if argumentos.mkt not in tendencias:
        log.error("El argumento de mercado no es válido: %s", argumentos.mkt)
        sys.exit()
        
    _environment = argumentos.env #'live'#practice
    id, token = Authenticate(environment=_environment).get_aunthenticate()
    api = API(access_token=token, environment=_environment)

    # Valores para market:
    # bull ==> mercado alcista. Compra y cierra posición
    # bear ==> mercado bajista. Vende y cierra posición

    _market = argumentos.mkt
    _granularity = argumentos.tmf

    # ================== SETUP =======================================
    _switch_rsi = {'M1': {'rsi': 14, 'upper': 70, 'lower': 30, 'count': 300},
                   'M5': {'rsi': 9, 'upper': 80, 'lower': 20, 'count': 5000},
                   'M15': {'rsi': 14, 'upper': 70, 'lower': 30, 'count': 100},
                   'M30': {'rsi': 14, 'upper': 70, 'lower': 30, 'count': 100},
                   'H1': {'rsi': 14, 'upper': 70, 'lower': 30, 'count': 500},
                   'H4': {'rsi': 14, 'upper': 70, 'lower': 30, 'count': 500},
                   'D': {'rsi': 14, 'upper': 70, 'lower': 30, 'count': 30}}

    _rsi_period = _switch_rsi.get(_granularity).get('rsi', 14)
    _rsi_upper = _switch_rsi.get(_granularity).get('upper', 70)
    _rsi_lower = _switch_rsi.get(_granularity).get('lower', 30)
    _count = _switch_rsi.get(_granularity).get('count', 300)

    # ========================================
    _switch_instruments = {"SP500":{"id":"SPX500_USD", "unt": 2},
                            "DE30":{"id":"DE30_EUR", "unt": 2},
                            "EURUSD":{"id": "EUR_USD", "unt": 60000}
    }

    _units = _switch_instruments.get(argumentos.inst.upper()).get("unt")
    _instrument = _switch_instruments.get(argumentos.inst.upper()).get("id")
    # ===============================

    log.info('Connected and authenticated for %s.\n %s\n %s\n %s\n %s\n %s \n %s', _environment, id, _market.upper(),
             _units, _instrument, _switch_rsi.get(_granularity), _granularity)

    _ichimoku = False

    ######################################################################
    # price = Prices(token=token, instruments='EUR_USD', id=id)
    price = Prices(token=token, instruments=_instrument, id=id, environment=_environment)

    if _ichimoku is True:
        data_frame_candles = price.get_candles_mid(granularity=_granularity, count=4000)
        # # log.info('Dataframe %s', data_frame_candles[-300:])
        ich = Ichimoku()
        df = ich.indicators(data_frame_candles)
    # log.info('Dataframe %s', df[-100:])
    # ================================================================

    # Obtain data from broker. Dataframe!
    df = price.get_candles_close(count=_count, granularity=_granularity)

    # Volume().indicators(df=df)

    # Backtest strategies
    for i in [20]:
        bollinguer = BollinguerRsiMacd(market=_market, units=_units, window=i)
        try:
            log.info(bollinguer.__class__.__name__)
            #bollinguer.indicators(df=df, timeperiod=9, rsi_upper=75, rsi_lower=25)
            bollinguer.indicators(df=df, timeperiod=_rsi_period, rsi_upper=_rsi_upper, rsi_lower=_rsi_lower)
        except Exception as err:
            log.error(err)
    # ======================================
    flag = RsiFlagStrategy(market=_market, units=_units, windows=_rsi_period)
    try:
        log.info(flag.__class__.__name__)
        flag.indicators(df, upper=_rsi_upper, lower=_rsi_lower)
    except Exception as err:
        log.error(err)
    # =====================================
    macd = MACDRSIStrategy(market=_market, units=_units, windows=_rsi_period)
    try:
        log.info(macd.__class__.__name__)
        #macd.indicators(df, upper=_rsi_upper, lower=_rsi_lower)
    except Exception as err:
        log.error(err)
    # ====================================
    rsi = RsiMomentumStrategy(market=_market, units=_units, windows=_rsi_period)
    try:
        log.info(rsi.__class__.__name__)
        rsi.indicators(df, upper=_rsi_upper, lower=_rsi_lower)
    except Exception as err:
        log.error(err)
    # ====================================
    adx = BbandsAdxStrategy(market=_market, units=_units, windows=_rsi_period)
    try:
        log.info(adx.__class__.__name__)
        #adx.indicators(df, rsi_lower=_rsi_lower, rsi_upper=_rsi_upper, time_period=_rsi_period)
    except Exception as err:
        log.error(err)
    # ====================================
    adx_rsi = AdxRsiStrategy(market=_market, units=_units, windows=_rsi_period)
    try:
        log.info(adx_rsi.__class__.__name__)
        #adx_rsi.indicators(df, lower=_rsi_lower, upper=_rsi_upper, time_period=_rsi_period)
    except Exception as err:
        log.error(err)

    # scalping = Scalping(market=market, units=units, window=9)
    # try:
    #     scalping.indicators(df)
    # except Exception as err:
    #     log.error(err)
