import datetime
import logging
import requests
import urllib3
import pandas as pd

requests.packages.urllib3.util.ssl_.DEFAULT_CIPHERS = "ALL:@SECLEVEL=1"

class Prices(object):
    def __init__(self, token, instruments, id, environment='practice'):
        logging.basicConfig(level='DEBUG')
        self.log = logging.getLogger(__name__)
        self.token=token
        self.instruments=instruments
        self.id=id

        if environment == 'practice':
            self.url_candles = 'https://api-fxpractice.oanda.com/v3/instruments/'
        else:
            self.url_candles = 'https://api-fxtrade.oanda.com/v3/instruments/'

    def get_candles_mid(self, granularity='D', price='MBA', count=5000):
        headers = {'Authorization': 'Bearer ' + self.token}

        url_candle = self.url_candles + self.instruments + '/candles'
        candles = requests.request('GET', url=url_candle, headers=headers,
                                   params={'count': count, 'price': price, 'granularity': granularity})

        self.log.debug(candles.json())

        lsIndex = []
        lsAsk = []
        lsBid = []
        lsMid = []
        lsVolume = []

        for js in candles.json()['candles']:
            lsIndex.append(datetime.datetime.strptime(js.get('time')[:19], '%Y-%m-%dT%H:%M:%S'))
            # lsIndex.append(R.get('time'))
            lsAsk.append(float(js.get('mid').get('h')))
            lsBid.append(float(js.get('mid').get('l')))
            lsMid.append(float(js.get('mid').get('c')))
            lsVolume.append(js.get('volume'))

        self.data_frame_candles = pd.DataFrame(
            data={'date': lsIndex, 'h': lsAsk, 'l': lsBid, 'mid':lsMid, 'volume': lsVolume},
            columns=['date', 'h', 'l', 'mid', 'volume'])
        # self.data_frame_candles= self.data_frame_candles.set_index(pd.DatetimeIndex(self.data_frame_candles['date']))

        self.log.debug('Candles... \n%s', self.data_frame_candles)

        return self.data_frame_candles

    def get_candles_close(self, granularity='M1', price='MBA', count=5000):
        headers = {'Authorization': 'Bearer ' + self.token}

        url_candle = self.url_candles + self.instruments + '/candles'
        candles = requests.request('GET', url=url_candle, headers=headers,
                                   params={'count': count, 'price': price, 'granularity': granularity})

        self.log.debug(candles.json())

        if 'errorMessage' in candles.json():
            self.log.warning(candles.json()['errorMessage'])

        lsIndex = []
        lsAsk = []
        lsBid = []
        lsMid = []
        lsMidH = []
        lsMidL = []
        lsVolume = []

        for js in candles.json()['candles']:
            lsIndex.append(datetime.datetime.strptime(js.get('time')[:19], '%Y-%m-%dT%H:%M:%S'))
            # lsIndex.append(R.get('time'))
            lsAsk.append(float(js.get('ask').get('c')))
            lsBid.append(float(js.get('bid').get('c')))
            lsMid.append(float(js.get('mid').get('c')))
            lsMidH.append(float(js.get('mid').get('h')))
            lsMidL.append(float(js.get('mid').get('l')))
            lsVolume.append(js.get('volume'))

        self.data_frame_candles = pd.DataFrame(
            data={'date': lsIndex, 'ask': lsAsk, 'bid': lsBid, 'mid': lsMid, 'mid_h':lsMidH, 'mid_l':lsMidL, 'volume': lsVolume},
            columns=['date', 'ask', 'bid', 'mid', 'mid_h', 'mid_l', 'volume'])
        # self.data_frame_candles= self.data_frame_candles.set_index(pd.DatetimeIndex(self.data_frame_candles['date']))

        self.log.debug('Candles... \n%s', self.data_frame_candles)

        #self.data_frame_candles= pd.concat([self.data_frame_candles, self.streamPrice()], ignore_index=True, sort=False)

        return self.data_frame_candles

    def streamPrice(self):
        param = {"instruments": self.instruments}

        headers = {'Authorization': 'Bearer ' + self.token}

        # url_stream = 'https://stream-fxpractice.oanda.com/v3/accounts/' + self.id + '/pricing/stream'
        # obtiene precio actual...
        url_price = 'https://api-fxpractice.oanda.com/v3/accounts/' + self.id + '/pricing'

        response = requests.request('GET', url=url_price, headers=headers, params=param)
        dict = []
        for r in response.json()['prices']:
            #self.log.info(r)
            time_ = datetime.datetime.strptime(r.get('time')[:19], '%Y-%m-%dT%H:%M:%S')
            bid = float(r.get('bids')[0].get('price'))
            ask = float(r.get('asks')[0].get('price'))
            dict = {'date': [time_], 'ask': [ask], 'bid': [bid]}

        data_frame_stream = pd.DataFrame(data=dict, columns=['date', 'ask', 'bid'])

        print(data_frame_stream.tail())

        return data_frame_stream

    def streaming_price(self):
        param = {"instruments": self.instruments}

        headers = {'Authorization': 'Bearer ' + self.token}

        url_stream='https://stream-fxpractice.oanda.com/v3/accounts/'+self.id+'/pricing/stream'

        sesion = requests.Session()
        r=requests.request('GET', url=url_stream, headers=headers, params=param)
        p=r.prepare()
        resp = sesion.send(p, stream=True, verify=True)

        print(resp.json())
