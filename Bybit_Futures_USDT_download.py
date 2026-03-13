import requests
import pandas as pd
import time
import os
import sys
import logging
import math
from datetime import datetime

# ================== НАСТРОЙКИ ==================

EXCHANGE = "BYBIT"
CATEGORY = "linear"
INTERVAL = "1"
LIMIT = 1000                 # максимум Bybit
RAW_DIR = "Raw_Data"
LOG_DIR = "Logs"

KLINE_URL = "https://api.bybit.com/v5/market/kline"
SYMBOLS_URL = "https://api.bybit.com/v5/market/instruments-info"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ================== ЛОГИРОВАНИЕ ==================

log_file = os.path.join(
    LOG_DIR,
    f"bybit_downloader_{datetime.utcnow().date()}.log"
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)

log = logging.info

# ================== УТИЛИТЫ ==================

def clear_console():
    os.system("cls" if os.name == "nt" else "clear")


def aligned_now_ts_ms() -> int:
    """Текущее время, выровненное по минуте, мс = 000"""
    now_sec = int(time.time())
    now_sec -= now_sec % 60
    return now_sec * 1000


def get_linear_usdt_symbols():
    log("Получаем список торговых пар Bybit (linear USDT)")
    symbols = []
    cursor = None

    while True:
        params = {"category": CATEGORY, "limit": 1000}
        if cursor:
            params["cursor"] = cursor

        r = requests.get(SYMBOLS_URL, params=params, timeout=20).json()
        if r["retCode"] != 0:
            raise RuntimeError(r["retMsg"])

        for item in r["result"]["list"]:
            if item["quoteCoin"] == "USDT":
                symbols.append(item["symbol"])

        cursor = r["result"].get("nextPageCursor")
        if not cursor:
            break

    log(f"Найдено торговых пар: {len(symbols)}")
    return symbols


def file_already_exists(symbol: str) -> bool:
    prefix = f"{EXCHANGE}_{symbol}_"
    return any(f.startswith(prefix) for f in os.listdir(RAW_DIR))


def safe_kline_request(params, symbol, start_ts, end_ts, batch, retry_delay=5):
    """Безопасный запрос с автоповтором при сетевых и API-ошибках"""
    while True:
        try:
            r = requests.get(KLINE_URL, params=params, timeout=20)
            r.raise_for_status()
            data = r.json()

            if data.get("retCode") != 0:
                log(f"{symbol} | API ошибка: {data.get('retMsg')} — повтор через {retry_delay}с")
                time.sleep(retry_delay)
                continue

            return data["result"]["list"]

        except requests.exceptions.RequestException as e:
            log(
                f"{symbol} | Ошибка соединения "
                f"({pd.to_datetime(start_ts, unit='ms')} → {pd.to_datetime(end_ts, unit='ms')}, "
                f"batch={batch}) : {e} — повтор через {retry_delay}с"
            )
            time.sleep(retry_delay)

        except ValueError as e:
            log(
                f"{symbol} | Ошибка JSON "
                f"({pd.to_datetime(start_ts, unit='ms')} → {pd.to_datetime(end_ts, unit='ms')}, "
                f"batch={batch}) : {e} — повтор через {retry_delay}с"
            )
            time.sleep(retry_delay)

# ================== ЗАГРУЗКА ==================

def download_symbol(symbol: str):
    log(f"▶ Старт загрузки {symbol}")

    end_ts = aligned_now_ts_ms()
    all_candles = []
    batch = LIMIT

    while True:
        clear_console()

        start_ts = end_ts - (batch - 1) * 60_000

        log(
            f"{symbol} | batch={batch} | "
            f"{pd.to_datetime(start_ts, unit='ms')} → "
            f"{pd.to_datetime(end_ts, unit='ms')}"
        )

        params = {
            "category": CATEGORY,
            "symbol": symbol,
            "interval": INTERVAL,
            "start": start_ts,
            "end": end_ts,
            "limit": batch
        }

        candles = safe_kline_request(params, symbol, start_ts, end_ts, batch)

        # --- НЕТ ДАННЫХ ---
        if not candles:
            if batch > 1:
                batch = math.ceil(batch / 2)
                log(f"{symbol} | данных нет → уменьшаем batch до {batch}")
                continue
            else:
                log(f"■ Истинный конец истории для {symbol}")
                break

        # --- ДАННЫЕ ЕСТЬ ---
        all_candles.extend(candles)

        df_tmp = pd.DataFrame(candles, columns=[
            "timestamp", "open", "high", "low",
            "close", "volume", "turnover"
        ])
        df_tmp["timestamp"] = pd.to_datetime(df_tmp["timestamp"].astype("int64"), unit="ms")

        earliest = df_tmp["timestamp"].min()

        # КЛЮЧЕВОЕ МЕСТО: сдвиг без перекрытий
        end_ts = int(earliest.timestamp() * 1000) - 60_000

        batch = LIMIT

    if not all_candles:
        log(f"✗ Нет данных для {symbol}")
        return

    # ================== ОБРАБОТКА ==================

    df = pd.DataFrame(all_candles, columns=[
        "timestamp", "open", "high", "low",
        "close", "volume", "turnover"
    ])

    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = pd.to_numeric(df[col])

    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms")
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])

    start_year = df["timestamp"].dt.year.min()
    end_year = df["timestamp"].dt.year.max()

    filename = f"{EXCHANGE}_{symbol}_LINEAR_{start_year}_{end_year}.csv"
    path = os.path.join(RAW_DIR, filename)

    df.to_csv(path, index=False)

    log(f"✓ {symbol} сохранён")
    log(f"  Свечей: {len(df)}")
    log(f"  Период: {df['timestamp'].min()} → {df['timestamp'].max()}")
    log("-" * 90)

# ================== MAIN ==================

def main():
    symbols = ["DOGEUSDT"] #get_linear_usdt_symbols()

    for symbol in symbols:
        if file_already_exists(symbol):
            log(f"⏩ {symbol} уже существует — пропуск")
            continue

        try:
            download_symbol(symbol)
        except Exception as e:
            log(f"✗ Ошибка {symbol}: {e}")

if __name__ == "__main__":
    main()
