import os
import sys
import glob
import logging
import platform
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, ROCIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, AccDistIndexIndicator, ChaikinMoneyFlowIndicator

# === Логирование ===
def setup_logger(log_file):
    logger = logging.getLogger('trade_data_logger')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_file, mode='a')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

# === Очистка консоли ===
def clear_console():
    if platform.system() == "Windows":
        os.system('cls')
    else:
        os.system('clear')

# === Индикаторы ===
def calc_ema(df, period, col='close'):
    return EMAIndicator(close=df[col], window=period).ema_indicator()

def calc_rsi(df, period=14, col='close'):
    return RSIIndicator(close=df[col], window=period).rsi()

def calc_macd(df, col='close'):
    macd = MACD(close=df[col])
    return macd.macd(), macd.macd_signal(), macd.macd_diff()

def calc_stoch(df):
    stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
    return stoch.stoch(), stoch.stoch_signal()

def calc_williams_r(df, period=14):
    return WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close'], lbp=period).williams_r()

def calc_roc(df, period=12, col='close'):
    return ROCIndicator(close=df[col], window=period).roc()

def calc_bollinger(df, period=20, col='close'):
    bb = BollingerBands(close=df[col], window=period)
    return bb.bollinger_hband(), bb.bollinger_lband(), bb.bollinger_mavg()

def calc_atr(df, period=14):
    return AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=period).average_true_range()

def calc_obv(df):
    return OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()

def calc_adi(df):
    return AccDistIndexIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume']).acc_dist_index()

def calc_cmf(df, period=20):
    return ChaikinMoneyFlowIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'], window=period).chaikin_money_flow()

# === Ресемплирование ===
def resample_df(df, timeframe='5T'):
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'turnover': 'sum'
    }
    return df.resample(timeframe).agg(ohlc_dict).dropna()

# === Процентное изменение close ===
def calc_prev_close_pct(df):
    return df['close'].shift(1) / df['close']

# === Удаление строк с NaN в начале ===
def drop_initial_na(df):
    first_valid = df.dropna().index[0]
    return df.loc[first_valid:]

# === Основная обработка файла ===
def process_file(filepath, logger):
    clear_console()
    logger.info(f"Обработка файла: {filepath}")

    # Пропуск файлов с '-'
    if '-' in os.path.basename(filepath):
        logger.info("Пропущен (квартальный фьючерс)")
        return

    # Чтение данных
    df = pd.read_csv(filepath, names=["datetime", "open", "high", "low", "close", "volume", "turnover"], header=None)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)
    logger.info(f"Загружено строк: {len(df)}")

    # Удалить последнюю строку
    df = df.iloc[:-1]
    logger.info(f"Удалена последняя строка. Осталось: {len(df)}")

    # Ресемплирование
    for tf in ['5T', '15T']:
        df_resampled = resample_df(df, tf)
        logger.info(f"Ресемплирование {tf}: {len(df_resampled)} строк")

        # Индикаторы
        df_resampled['ema10'] = calc_ema(df_resampled, 10)
        df_resampled['ema50'] = calc_ema(df_resampled, 50)
        df_resampled['ema100'] = calc_ema(df_resampled, 100)
        df_resampled['rsi'] = calc_rsi(df_resampled)
        df_resampled['macd'], df_resampled['macd_signal'], df_resampled['macd_diff'] = calc_macd(df_resampled)
        df_resampled['stoch'], df_resampled['stoch_signal'] = calc_stoch(df_resampled)
        df_resampled['williams_r'] = calc_williams_r(df_resampled)
        df_resampled['roc'] = calc_roc(df_resampled)
        df_resampled['bb_h'], df_resampled['bb_l'], df_resampled['bb_m'] = calc_bollinger(df_resampled)
        df_resampled['atr'] = calc_atr(df_resampled)
        df_resampled['obv'] = calc_obv(df_resampled)
        df_resampled['adi'] = calc_adi(df_resampled)
        df_resampled['cmf'] = calc_cmf(df_resampled)

        # Процентное изменение close
        df_resampled['prev_close_pct'] = calc_prev_close_pct(df_resampled)

        # Удалить строки с NaN в начале
        df_resampled = drop_initial_na(df_resampled)
        logger.info(f"После удаления NaN: {len(df_resampled)} строк")

        # Анализ периода данных
        min_date, max_date = df_resampled.index.min(), df_resampled.index.max()
        total_days = (max_date - min_date).days
        logger.info(f"Данные с {min_date} по {max_date} ({total_days} дней)")

        # Сохранение
        base_name = os.path.splitext(os.path.basename(filepath))[0]
        tf_str = tf.replace('T', 'm')
        if total_days < 547:  # 1.5 года = 365*1.5 = 547.5 дней
            out_path = f"Test_data/{base_name}_{tf_str}_Test.csv"
            os.makedirs("Test_data", exist_ok=True)
            df_resampled.to_csv(out_path)
            logger.info(f"Сохранено в {out_path}")
        else:
            # Последний год
            last_year_start = max_date - pd.Timedelta(days=365)
            df_last_year = df_resampled[df_resampled.index >= last_year_start]
            df_rest = df_resampled[df_resampled.index < last_year_start]

            os.makedirs("Test_data", exist_ok=True)
            os.makedirs("Train_data", exist_ok=True)

            out_test = f"Test_data/{base_name}_{tf_str}_Test.csv"
            out_train = f"Train_data/{base_name}_{tf_str}_Train.csv"
            df_last_year.to_csv(out_test)
            df_rest.to_csv(out_train)
            logger.info(f"Сохранено в {out_test} (последний год)")
            logger.info(f"Сохранено в {out_train} (старые данные)")

def main():
    log_file = "trade_data_processing.log"
    logger = setup_logger(log_file)

    raw_data_path = "Raw_Data"
    files = glob.glob(os.path.join(raw_data_path, "*.csv"))
    logger.info(f"Найдено файлов: {len(files)}")

    for file in files:
        try:
            process_file(file, logger)
        except Exception as e:
            logger.error(f"Ошибка при обработке {file}: {e}")

if __name__ == "__main__":
    main()