import yfinance as yf
import pandas as pd
import ta
import threading
import time
import asyncio
import json
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional, Any
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

# ==== 분석 함수 ====
def get_data(symbol: str) -> pd.DataFrame:
    df = yf.download(symbol, period="10d", interval="1m", progress=False)
    return df

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Close" not in df.columns:
        return df
    df["rsi"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()
    df["macd"] = ta.trend.MACD(df["Close"]).macd()
    df["macd_signal"] = ta.trend.MACD(df["Close"]).macd_signal()
    df["ma20"] = df["Close"].rolling(20).mean()
    df["bb_upper"] = ta.volatility.BollingerBands(df["Close"]).bollinger_hband()
    df["bb_lower"] = ta.volatility.BollingerBands(df["Close"]).bollinger_lband()
    return df

def detect_signals(df: pd.DataFrame) -> Dict[str, List[str]]:
    result = {}
    for idx, row in df.iterrows():
        sigs = []
        # RSI
        if row.get("rsi", 50) > 70:
            sigs.append("과열 (RSI>70)")
        elif row.get("rsi", 50) < 30:
            sigs.append("과매도 (RSI<30)")
        # MACD
        if row.get("macd", 0) > row.get("macd_signal", 0):
            sigs.append("MACD 골든크로스")
        elif row.get("macd", 0) < row.get("macd_signal", 0):
            sigs.append("MACD 데드크로스")
        # MA20 돌파/이탈
        close = row.get("Close", 0)
        ma20 = row.get("ma20", 0)
        if pd.notna(ma20):
            if close > ma20:
                sigs.append("20MA 돌파 (매수)")
            elif close < ma20:
                sigs.append("20MA 이탈 (매도)")
        # 볼밴 상하단
        if close > row.get("bb_upper", 0):
            sigs.append("볼린저밴드 상단 이탈")
        if close < row.get("bb_lower", 0):
            sigs.append("볼린저밴드 하단 이탈")
        if sigs:
            result[str(idx)] = sigs
    return result

def backtest_signals(df: pd.DataFrame, signals: Dict[str, List[str]]) -> pd.DataFrame:
    result = []
    df = df.copy()
    for idx in signals.keys():
        i = df.index.get_loc(pd.Timestamp(idx))
        entry = df.iloc[i]
        entry_price = entry["Close"]
        entry_time = idx
        signal_list = signals[idx]
        # 단순: 5분 후 청산, 수익률 계산
        exit_i = i+5 if i+5 < len(df) else len(df)-1
        exit = df.iloc[exit_i]
        exit_price = exit["Close"]
        exit_time = str(exit.name)
        ret = (exit_price - entry_price) / entry_price if entry_price else 0
        result.append({
            "signal_time": entry_time,
            "signals": signal_list,
            "entry_price": entry_price,
            "exit_time": exit_time,
            "exit_price": exit_price,
            "ret": ret
        })
    return pd.DataFrame(result)

def analyze_timeframes(symbol: str) -> List[str]:
    intervals = [("1m", "1일"), ("5m", "5일"), ("15m", "10일")]
    reports = []
    for interval, label in intervals:
        try:
            df = yf.download(symbol, period="5d", interval=interval, progress=False)
            df = calculate_indicators(df)
            sig = detect_signals(df)
            last_dt = str(df.index[-1]) if not df.empty else "?"
            if sig:
                last_sig = list(sig.values())[-1]
                reports.append(f"[{label}] {','.join(last_sig)} ({last_dt})")
            else:
                reports.append(f"[{label}] 신호 없음")
        except Exception as e:
            reports.append(f"[{label}] 에러: {e}")
    return reports

def is_valid_data(df):
    return (
        df is not None and not df.empty
        and "Close" in df.columns
        and df['Close'].notnull().any()
        and df['Close'].mean() > 0
    )

# ==== 서버, 캐시, 쓰레드, 관리자 ====
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

MAX_WATCH = 5
data_cache: Dict[str, dict] = {}
watched_symbols: set = set()
symbol_threads: Dict[str, threading.Thread] = {}

def update_cache(symbol):
    global data_cache, watched_symbols, symbol_threads
    print(f"[update_cache] 쓰레드 시작: {symbol}")
    error_count = 0
    max_error_count = 3
    last_cache = None
    while symbol in watched_symbols:
        print(f"[update_cache] {symbol}: get_data() 호출")
        try:
            df = get_data(symbol)
            print(f"[update_cache] {symbol}: df shape: {df.shape}")
            if not is_valid_data(df):
                print(f"[update_cache] {symbol}: 데이터 유효하지 않음")
                error_count += 1
                time.sleep(3)
                continue
            error_count = 0
            df = calculate_indicators(df)
            signal_dict = detect_signals(df)
            backtest_result = backtest_signals(df, signal_dict)
            mtf_report = analyze_timeframes(symbol)
            latest_idx = df.index[-1]
            latest_row = df.iloc[-1]
            current_cache = {
                "symbol": symbol,
                "latest_price": float(latest_row['Close']),
                "latest_date": str(latest_idx),
                "signals": signal_dict.get(str(latest_idx), []),
                "backtest": backtest_result.to_dict(orient='records'),
                "mtf_report": mtf_report,
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            # 변경 사항이 있을 때만 data_cache 갱신
            if current_cache != last_cache:
                data_cache[symbol] = current_cache
                last_cache = current_cache
        except Exception as e:
            error_count += 1
        if error_count >= max_error_count:
            watched_symbols.discard(symbol)
            symbol_threads.pop(symbol, None)
            data_cache.pop(symbol, None)
            break
        time.sleep(10)

def start_symbol_thread(symbol):
    if symbol in watched_symbols:
        return
    if len(watched_symbols) >= MAX_WATCH:
        raise HTTPException(429, f"최대 {MAX_WATCH}개 심볼만 감시 가능합니다")
    watched_symbols.add(symbol)
    t = threading.Thread(target=update_cache, args=(symbol,), daemon=True)
    symbol_threads[symbol] = t
    t.start()

# ==== REST API ====
@app.get("/monitor", response_model=TradingSignalResponse)
def monitor(symbol: str = Query("TQQQ")):
    start_symbol_thread(symbol)
    if symbol in data_cache:
        return data_cache[symbol]
    return {
        "symbol": symbol, "latest_price": 0.0, "latest_date": "",
        "signals": [], "backtest": [], "mtf_report": ["데이터 없음 또는 잘못된 심볼"], "last_updated": ""
    }

@app.get("/monitors")
def monitors():
    return {
        "watched_symbols": list(watched_symbols),
        "active_threads": list(symbol_threads.keys()),
        "cache_symbols": list(data_cache.keys())
    }

@app.post("/monitor/add")
def add_symbol(symbol: str):
    start_symbol_thread(symbol)
    return {"result": "added", "watched": list(watched_symbols)}

@app.delete("/monitor/remove")
def remove_symbol(symbol: str):
    watched_symbols.discard(symbol)
    symbol_threads.pop(symbol, None)
    data_cache.pop(symbol, None)
    return {"result": "removed", "watched": list(watched_symbols)}

# ==== WebSocket 관리 ====
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
    try:
        start_symbol_thread(symbol)
        # 1. 최초 연결시 현재 데이터가 있으면 바로 전송
        if symbol in data_cache:
            await websocket.send_text(json.dumps(data_cache[symbol]))
        last_sent = json.dumps(data_cache[symbol]) if symbol in data_cache else None
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
