ASSET_TRADES = {
            'EURUSD': 0, 'GBPUSD': 0, 'USDCAD': 0, 'USDJPY': 0, 'AUDUSD': 0, 'USDCHF': 0, 'NZDUSD': 0,
            'GBPCHF': 0, 'CADCHF': 0, 'AUDCAD': 0, 'NZDCHF': 0, 'NZDJPY': 0, 'AUDJPY': 0, 'GBPJPY': 0,
            'EURJPY': 0, 'EURCHF': 0, 'GBPCAD': 0, 'EURCAD': 0, 'SPX500USD':0, 'NAS100USD':0, 'US30USD':0,
            'DE30EUR' :0, 'UK100GBP' :0, 'XAUUSD' :0, 'XAGUSD' :0, 'WTICOUSD' :0, 'AUDCHF':0
        }



GOLD = ['XAUUSD']
INDICES = ['SPX500USD','US30USD','NAS100USD','UK100GBP','DE30EUR']
JPY_PAIRS = ['USDJPY','GBPJPY','NZDJPY','AUDJPY','EURJPY','XAGUSD','WTICOUSD']
OTHER_PAIRS = ['EURUSD' , 'GBPUSD', 'USDCAD', 'AUDUSD', 'USDCHF', 'NZDUSD', 'GBPCHF', 'CADCHF', 'AUDCHF', 'NZDCHF','EURCHF', 'GBPCAD', 'EURCAD', 'EURGBP', 'EURAUD','EURNZD','GBPAUD','GBPNZD','AUDCAD','AUDNZD' ,'NZDCAD'] 
PIP_VALUES = {
    "JPY_PAIRS": 0.01,
    "INDICES": 1,
    "GOLD": 0.1,
    "OTHER_PAIRS": 0.0001
}

DEFAULT_KEYS = [
        "symbol", "runtime", "current_bias", "ICTS_5P_DT","ICTS_5P_Dir", "ICTS_O", "ICTS_H", "ICTS_L", 
        "ICTS_C", "bias_entry_alignment", "trade_execution", "atr_d1", "order_dt", 
        "order_no", "order_type", "order_action", "base_price", "order_price", 
        "stop_loss", "tp", "lot", "bid", "ask", "spread", "stop_loss_break_even",
        "SL_$","TP_$"
    ]

USD_PIP_VALUE = 1.00
CAD_PIP_VALUE = 0.69
JPY_PIP_VALUE = 0.63
CHF_PIP_VALUE = 1.09

ASSETS = { 
        'EURUSD' : 'EURUSD.sd',
         'GBPUSD' :'GBPUSD.sd',
         'USDCAD' :'USDCAD.sd',
         'USDJPY' :'USDJPY.sd',
         'AUDUSD' :'AUDUSD.sd',
         'USDCHF' :'USDCHF.sd',
         'NZDUSD' :'NZDUSD.sd',
         'GBPCHF' :'GBPCHF.sd',
         'CADCHF' :'CADCHF.sd',
         'AUDCHF' :'AUDCHF.sd',
         'NZDCHF' :'NZDCHF.sd',
         'NZDJPY' :'NZDJPY.sd',
         'AUDJPY' : 'AUDJPY.sd',
         'GBPJPY' :'GBPJPY.sd',
         'EURJPY' :'EURJPY.sd',
         'EURCHF' :'EURCHF.sd',
         'GBPCAD' :'GBPCAD.sd',
         'EURCAD' :'EURCAD.sd',

         'EURGBP' :'EURGBP.sd',
         'EURAUD' :'EURAUD.sd',
         'EURNZD' :'EURNZD.sd',
         'GBPAUD' :'GBPAUD.sd',
         'GBPNZD' :'GBPNZD.sd',
         'AUDCAD' :'AUDCAD.sd',
         'AUDNZD' :'AUDNZD.sd',
         'NZDCAD': 'NZDCAD.sd',

         'SPX500USD' :'US500Roll',
         'NAS100USD' :'UT100Roll',
         'US30USD' :'US30Roll',
         'DE30EUR' :'DE40Roll',
         'UK100GBP' :'UK100Roll',
         'XAUUSD' :'XAUUSD.sd',
         'XAGUSD' :'XAGUSD.sd',
         'WTICOUSD' :'USOILRoll'
         }


AHMED_ASSET = { 'EURUSD' : 'EURUSD.sd',
         'GBPUSD' :'GBPUSD.sd',
         'USDCAD' :'USDCAD.sd',
         'USDJPY' :'USDJPY.sd',
         'AUDUSD' :'AUDUSD.sd',
         'USDCHF' :'USDCHF.sd',
         'NZDUSD' :'NZDUSD.sd',
         'GBPCHF' :'GBPCHF.sd',
         'CADCHF' :'CADCHF.sd',
         'AUDCHF' :'AUDCHF.sd',
         'NZDCHF' :'NZDCHF.sd',
         'NZDJPY' :'NZDJPY.sd',
         'AUDJPY' : 'AUDJPY.sd',
         'GBPJPY' :'GBPJPY.sd',
         'EURJPY' :'EURJPY.sd',
         'EURCHF' :'EURCHF.sd',
         'GBPCAD' :'GBPCAD.sd',
         'EURCAD' :'EURCAD.sd',
         'EURGBP' : 'EURGBP.sd',
         'EURAUD' : 'EURAUD.sd',
         'EURNZD' : 'EURNZD.sd',
         'GBPAUD' : 'GBPAUD.sd',
         'GBPNZD' : 'GBPNZD.sd',
         'AUDCAD' : 'AUDCAD.sd',
         'NZDCAD' : 'NZDCAD.sd',
         'AUDNZD' : 'AUDNZD.sd',
         'SPX500USD' :'US500Roll',
         'NAS100USD' :'UT100Roll',
         'US30USD' :'US30Roll',
         'DE30EUR' :'DE40Roll',
         'UK100GBP' :'UK100Roll',
         'XAUUSD' :'XAUUSD.sd',
         'XAGUSD' :'XAGUSD.sd',
         'WTICOUSD' :'USOILRoll'
         }