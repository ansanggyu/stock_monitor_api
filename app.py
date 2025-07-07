import yfinance as yf
import pandas as pd
import ta
import threading
import time
import json
import asyncio
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ====== 데이터 타입 ======

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

# ====== FastAPI 세팅 ======

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== 전역 데이터 ======
data_cache: Dict[str, Any] = {}
watched_symbols = set()
symbol_threads = {}
CACHE_INTERVAL = 1  # 초

# ====== 유틸 ======

def is_valid_data(df: pd.DataFrame) -> bool:
    return (
        df is not None and not df.empty
        and "Close" in df.columns
        and df['Close'].notnull().any()
        and df['Close'].mean() > 0
    )

def update_cache(symbol: str):
    """
    각 심볼마다 쓰레드로 동작. yfinance -> signal 분석 -> cache 갱신.
    """
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

def start_symbol_thread(symbol: str):
    """
    해당 symbol 데이터 감시 쓰레드가 없으면 새로 시작.
    """
    if symbol in watched_symbols:
        return
    watched_symbols.add(symbol)
    t = threading.Thread(target=update_cache, args=(symbol,), daemon=True)
    symbol_threads[symbol] = t
    t.start()

# ====== REST API 엔드포인트 ======

@app.get("/monitor", response_model=TradingSignalResponse)
def monitor(symbol: str = Query("TQQQ")):
    """
    특정 심볼의 최신 신호/데이터를 반환.
    """
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

# ====== WebSocket ConnectionManager ======

class ConnectionManager:
    """
    심볼별 접속 관리.
    """
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, symbol: str):
        await websocket.accept()
        self.active_connections.setdefault(symbol, []).append(websocket)

    def disconnect(self, websocket: WebSocket, symbol: str):
        if symbol in self.active_connections:
            try:
                self.active_connections[symbol].remove(websocket)
            except ValueError:
                pass
            if not self.active_connections[symbol]:
                del self.active_connections[symbol]

    async def broadcast(self, symbol: str, data: dict):
        for ws in list(self.active_connections.get(symbol, [])):
            try:
                await ws.send_text(json.dumps(data))
            except Exception:
                self.disconnect(ws, symbol)

manager = ConnectionManager()

# ====== WebSocket 엔드포인트 ======

@app.websocket("/ws/monitor")
async def websocket_endpoint(websocket: WebSocket):
    # symbol 파라미터 직접 파싱
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
