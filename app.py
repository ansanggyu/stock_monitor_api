import yfinance as yf
import pandas as pd
import ta
import threading
import time
import json
import asyncio

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict
from pydantic import BaseModel

# ==== 데이터 타입 정의 ====

class BacktestEntry(BaseModel):
    signal_time: str
    signals: List[str]
    entry_price: float
    exit_time: str
    exit_price: float
    ret: float

class TradingSignalResponse(BaseModel):
    symbol: str
    latest_price: float
    latest_date: str
    signals: List[str]
    backtest: List[BacktestEntry]
    mtf_report: List[str]
    last_updated: str

# ==== 데이터 관리 ====

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

data_cache = {}
watched_symbols = set()
symbol_threads = {}
CACHE_INTERVAL = 5  # 초 (API 과다호출 방지, 실전 배포시 1~5초 권장)

# ==== 데이터 함수 정의 ====

def get_data(symbol, interval="1m", period="5d"):
    """
    yfinance에서 데이터 불러오기 (일반적 1분봉, 5일치)
    """
    df = yf.download(
        tickers=symbol,
        interval=interval,
        period=period,
        progress=False,
        auto_adjust=True,
        threads=False,
    )
    if not df.empty:
        df.index = pd.to_datetime(df.index)
    return df

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    ta-lib 없이 pandas+ta 패키지로 주요 지표 추가
    """
    df = df.copy()
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["20MA"] = df["Close"].rolling(window=20).mean()
    bb = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
    df["BB_upper"] = bb.bollinger_hband()
    df["BB_lower"] = bb.bollinger_lband()
    return df

def detect_signals(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    날짜별 신호 판별 (RSI, MACD, 볼린저밴드 등)
    """
    signals = {}
    for i in range(1, len(df)):
        idx = df.index[i]
        row, prev = df.iloc[i], df.iloc[i - 1]
        sigs = []

        # 20MA 돌파/이탈
        if prev["Close"] < prev["20MA"] and row["Close"] > row["20MA"]:
            sigs.append("20MA 돌파 (매수)")
        if prev["Close"] > prev["20MA"] and row["Close"] < row["20MA"]:
            sigs.append("20MA 이탈 (매도)")

        # MACD 골든/데드크로스
        if prev["MACD"] < prev["MACD_signal"] and row["MACD"] > row["MACD_signal"]:
            sigs.append("MACD 골든크로스 (매수)")
        if prev["MACD"] > prev["MACD_signal"] and row["MACD"] < row["MACD_signal"]:
            sigs.append("MACD 데드크로스 (매도)")

        # RSI 과열/과매도
        if row["RSI"] > 70:
            sigs.append("과열 (RSI>70)")
        if row["RSI"] < 30:
            sigs.append("과매도 (RSI<30)")

        # 볼린저밴드 상/하단 터치
        if row["Close"] >= row["BB_upper"]:
            sigs.append("볼린저밴드 상단 이탈")
        if row["Close"] <= row["BB_lower"]:
            sigs.append("볼린저밴드 하단 이탈")

        if sigs:
            signals[str(idx)] = sigs
    return signals

def backtest_signals(df: pd.DataFrame, signal_dict: Dict[str, List[str]]) -> pd.DataFrame:
    """
    간단 백테스트 (매수/매도 신호 후 5분 뒤 진입/청산 수익률 계산)
    """
    records = []
    for idx, sigs in signal_dict.items():
        i = df.index.get_loc(pd.to_datetime(idx))
        entry_price = df.iloc[i]["Close"]
        # 5분 뒤 청산(혹은 마지막 값)
        exit_i = min(i + 5, len(df) - 1)
        exit_price = df.iloc[exit_i]["Close"]
        exit_time = df.index[exit_i]
        ret = (exit_price - entry_price) / entry_price
        records.append({
            "signal_time": str(df.index[i]),
            "signals": sigs,
            "entry_price": float(entry_price),
            "exit_time": str(exit_time),
            "exit_price": float(exit_price),
            "ret": float(ret),
        })
    return pd.DataFrame(records)

def analyze_timeframes(symbol: str) -> List[str]:
    """
    다중 타임프레임 리포트 예시
    """
    report = []
    # 1분봉
    df_1m = get_data(symbol, interval="1m", period="2d")
    df_1m = calculate_indicators(df_1m)
    rsi_1m = df_1m["RSI"].iloc[-1] if not df_1m.empty else None
    if rsi_1m is not None:
        report.append(f"[1분] RSI {rsi_1m:.2f} {'과열' if rsi_1m > 70 else '과매도' if rsi_1m < 30 else ''}")

    # 5분봉
    df_5m = get_data(symbol, interval="5m", period="5d")
    df_5m = calculate_indicators(df_5m)
    rsi_5m = df_5m["RSI"].iloc[-1] if not df_5m.empty else None
    if rsi_5m is not None:
        report.append(f"[5분] RSI {rsi_5m:.2f} {'과열' if rsi_5m > 70 else '과매도' if rsi_5m < 30 else ''}")

    # 1시간봉
    df_1h = get_data(symbol, interval="60m", period="30d")
    df_1h = calculate_indicators(df_1h)
    rsi_1h = df_1h["RSI"].iloc[-1] if not df_1h.empty else None
    if rsi_1h is not None:
        report.append(f"[1시간] RSI {rsi_1h:.2f} {'과열' if rsi_1h > 70 else '과매도' if rsi_1h < 30 else ''}")

    return report

def is_valid_data(df):
    return (
        df is not None and not df.empty
        and "Close" in df.columns
        and df['Close'].notnull().any()
        and df['Close'].mean() > 0
    )

def update_cache(symbol):
    global data_cache, watched_symbols, symbol_threads
    error_count = 0
    max_error_count = 5
    while True:
        try:
            df = get_data(symbol)
            if not is_valid_data(df):
                error_count += 1
                if error_count >= max_error_count:
                    watched_symbols.discard(symbol)
                    symbol_threads.pop(symbol, None)
                    data_cache.pop(symbol, None)
                    break
                time.sleep(3)
                continue
            error_count = 0
            df = calculate_indicators(df)
            signal_dict = detect_signals(df)
            backtest_result = backtest_signals(df, signal_dict)
            mtf_report = analyze_timeframes(symbol)
            latest_idx = df.index[-1]
            latest_row = df.iloc[-1]
            data_cache[symbol] = {
                "symbol": symbol,
                "latest_price": float(latest_row['Close']),
                "latest_date": str(latest_idx),
                "signals": signal_dict.get(str(latest_idx), []),
                "backtest": backtest_result.to_dict(orient='records'),
                "mtf_report": mtf_report,
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        except Exception as e:
            error_count += 1
            if error_count >= max_error_count:
                watched_symbols.discard(symbol)
                symbol_threads.pop(symbol, None)
                data_cache.pop(symbol, None)
                break
        time.sleep(CACHE_INTERVAL)

def start_symbol_thread(symbol):
    if symbol in watched_symbols:
        return
    watched_symbols.add(symbol)
    t = threading.Thread(target=update_cache, args=(symbol,), daemon=True)
    symbol_threads[symbol] = t
    t.start()

# ==== REST 엔드포인트 ====

@app.get("/monitor", response_model=TradingSignalResponse)
def monitor(symbol: str = Query("TQQQ")):
    start_symbol_thread(symbol)
    if symbol in data_cache:
        return data_cache[symbol]
    return {
        "symbol": symbol,
        "latest_price": 0.0,
        "latest_date": "",
        "signals": [],
        "backtest": [],
        "mtf_report": ["데이터 없음 또는 잘못된 심볼"],
        "last_updated": "",
    }

# ==== WebSocket 엔드포인트 ====

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, symbol: str):
        await websocket.accept()
        if symbol not in self.active_connections:
            self.active_connections[symbol] = []
        self.active_connections[symbol].append(websocket)

    def disconnect(self, websocket: WebSocket, symbol: str):
        if symbol in self.active_connections:
            self.active_connections[symbol].remove(websocket)
            if not self.active_connections[symbol]:
                del self.active_connections[symbol]

    async def broadcast(self, symbol: str, data: dict):
        if symbol in self.active_connections:
            for ws in list(self.active_connections[symbol]):
                try:
                    await ws.send_text(json.dumps(data))
                except Exception:
                    self.disconnect(ws, symbol)

manager = ConnectionManager()

@app.websocket("/ws/monitor")
async def websocket_endpoint(websocket: WebSocket):
    # 쿼리 파라미터에서 symbol 파싱
    symbol = websocket.query_params.get("symbol", "TQQQ")
    await manager.connect(websocket, symbol)
    start_symbol_thread(symbol)
    try:
        last_sent = None
        while True:
            await asyncio.sleep(1)
            if symbol in data_cache:
                current = json.dumps(data_cache[symbol], sort_keys=True)
                if current != last_sent:
                    await websocket.send_text(current)
                    last_sent = current
    except WebSocketDisconnect:
        manager.disconnect(websocket, symbol)
    except Exception:
        manager.disconnect(websocket, symbol)

# -------- END --------
