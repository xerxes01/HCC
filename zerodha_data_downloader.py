import logging
from kiteconnect import KiteConnect
import pandas as pd

logging.basicConfig(level=logging.WARNING)


api_key = "cv395he4v7cddp2e"
api_secret = "sya7vmshqw8m51r7x7s8bqztdi2al1wh"
historical_data_limits = {'minute':60, 'hour':365, 'day':2000, '3minute':90, '5minute':90, '10minute':90,
                          '15minute':180, '30minute':180, '60minute':365}


kite = KiteConnect(api_key= api_key)
kite.login_url()
data = kite.generate_session(request_token='Im8RT4xa1gns5tJvAXxCrbQ0DrELH6GE', api_secret=api_secret)
kite.set_access_token(data["access_token"])


class HistoricalData(object):
    
    def __init__(self, kite_id):
        self.kite_id = kite_id
    
    def get_data(self, start_date, end_date=None, interval='minute'):
        start_date = pd.to_datetime(start_date, dayfirst=True, format='%d/%m/%Y')
        if end_date is None:
            end_date = start_date + pd.Timedelta(days=historical_data_limits[interval])
        else:
            end_date = pd.to_datetime(end_date, dayfirst=True, format='%d/%m/%Y')
        data = kite.historical_data(self.kite_id, start_date, end_date, interval)
        return(data)
    
    def get_max_data(self, interval='minute'):
        tick_data = []
        start_date = pd.to_datetime('01/01/2015', dayfirst=True, format='%d/%m/%Y')
        while start_date < pd.datetime.now():
            end_date = start_date + pd.Timedelta(days=historical_data_limits[interval])
            data = kite.historical_data(self.kite_id, start_date, end_date, interval)
            tick_data.extend(data)
            start_date = end_date + pd.Timedelta(days=1)
        if tick_data:
            return(tick_data)
        else:
            pass


instruments = pd.DataFrame(kite.instruments())
instruments = instruments[(instruments['lot_size'] == 1) &
                          (instruments['expiry'] == '') &
                          (instruments['segment'] == 'NSE')]

for i in range(0, 380):
    download_class = HistoricalData(instruments.iloc[i, 0])
    data = pd.DataFrame(download_class.get_max_data())
    # Enter Path Here
    data.to_csv('')
    
    