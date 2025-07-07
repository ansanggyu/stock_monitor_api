import yfinance as yf
import pandas as pd
import ta
import threading
import time
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional, Any
from pydantic import BaseModel
import asyncio
import json

# ======== 데이터 타입 정의 ========
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

# ======== 분석 함수 (간단 예시, 실제 분석 함수는 이전 코드 참고) ========
def get_data(symbol: str) -> pd.DataFrame:
    df = yf.download(symbol, period="7d", interval="1m")
    return df

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # RSI, MACD 등 계산
    if "Close" not in df:
        return df
    df = df.copy()
    df["rsi"] = ta.momentum.rsi(df["Close"], window=14)
    macd = ta.trend.macd_diff(df["Close"])
    df["macd"] = macd
    df["20ma"] = df["Close"].rolling(window=20).mean()
    return df

def detect_signals(df: pd.DataFrame) -> Dict[str, List[str]]:
    signals = {}
    for i in range(1, len(df)):
        idx = df.index[i]
        row = df.iloc[i]
        row_prev = df.iloc[i-1]
        sigs = []
        if row["Close"] > row["20ma"] and row_prev["Close"] <= row_prev["20ma"]:
            sigs.append("20MA 돌파 (매수)")
        if row["Close"] < row["20ma"] and row_prev["Close"] >= row_prev["20ma"]:
            sigs.append("20MA 이탈 (매도)")
        if row["rsi"] is not None and row["rsi"] > 70:
            sigs.append("과열 (RSI>70)")
        if row["rsi"] is not None and row["rsi"] < 30:
            sigs.append("침체 (RSI<30)")
        if sigs:
            signals[str(idx)] = sigs
    return signals

def backtest_signals(df: pd.DataFrame, signal_dict: Dict[str, List[str]]) -> pd.DataFrame:
    # 매우 단순 예시: 각 신호 발생시점 진입가/청산가를 기록
    results = []
    for t, sigs in signal_dict.items():
        entry_idx = df.index.get_loc(pd.Timestamp(t))
        entry_price = df.iloc[entry_idx]["Close"]
        exit_idx = min(entry_idx + 5, len(df)-1)
        exit_price = df.iloc[exit_idx]["Close"]
        ret = (exit_price - entry_price) / entry_price
        results.append({
            "signal_time": str(df.index[entry_idx]),
            "signals": sigs,
            "entry_price": float(entry_price),
            "exit_time": str(df.index[exit_idx]),
            "exit_price": float(exit_price),
            "ret": float(ret)
        })
    return pd.DataFrame(results)

def analyze_timeframes(symbol: str) -> List[str]:
    # 다중타임프레임 보고서 예시 (임의)
    return [f"[{symbol}] 1분봉 강세", f"[{symbol}] 5분봉 약세"]

# ======== 서버/캐시/쓰레드 관리 ========
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
CACHE_INTERVAL = 10  # 초

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
    while symbol in watched_symbols:
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

def stop_symbol_thread(symbol):
    watched_symbols.discard(symbol)
    if symbol in symbol_threads:
        symbol_threads.pop(symbol, None)
    if symbol in data_cache:
        data_cache.pop(symbol, None)

# ======== REST API ========
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

@app.get("/monitors")
def monitors():
    # 현재 감시중인 심볼 전체 반환 (관리자 패널 용)
    return {"watched_symbols": list(watched_symbols), "active_threads": list(symbol_threads.keys()), "cache_symbols": list(data_cache.keys())}

@app.post("/monitor/add")
def add_monitor(symbol: str):
    start_symbol_thread(symbol)
    return {"ok": True, "msg": f"Added {symbol}"}

@app.delete("/monitor/remove")
def remove_monitor(symbol: str):
    stop_symbol_thread(symbol)
    return {"ok": True, "msg": f"Removed {symbol}"}

# ======== WebSocket 엔드포인트 ========
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
async def websocket_endpoint(websocket: WebSocket, symbol: str = "TQQQ"):
    await manager.connect(websocket, symbol)
    start_symbol_thread(symbol)
    try:
        last_sent = None
        while True:
            await asyncio.sleep(2)
            if symbol in data_cache:
                current = json.dumps(data_cache[symbol], sort_keys=True)
                if current != last_sent:
                    await websocket.send_text(current)
                    last_sent = current
    except WebSocketDisconnect:
        manager.disconnect(websocket, symbol)
    except Exception:
        manager.disconnect(websocket, symbol)
