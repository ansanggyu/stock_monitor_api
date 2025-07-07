import yfinance as yf
import pandas as pd
import ta
import threading
import time
from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Optional, Any
from pydantic import BaseModel

# ==== 데이터 타입 정의 (기존과 동일) ====

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

# === 기존 분석 함수 (get_data 등) 모두 그대로 복붙 ===
# (여기에 생략, 그대로 두세요)

# ==== 캐시, 쓰레드 관리 (기존과 동일) ====

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
CACHE_INTERVAL = 1  # 초

def is_valid_data(df):
    # (기존 그대로)
    return (
        df is not None and not df.empty
        and "Close" in df.columns
        and df['Close'].notnull().any()
        and df['Close'].mean() > 0
    )

def update_cache(symbol):
    # (기존 그대로)
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

# ==== 기존 REST 엔드포인트 (유지) ====
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

# ==== WebSocket 엔드포인트 추가 ====
import json

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
                    # 연결 끊긴 경우 정리
                    self.disconnect(ws, symbol)

manager = ConnectionManager()

@app.websocket("/ws/monitor")
async def websocket_endpoint(websocket: WebSocket, symbol: str = "TQQQ"):
    # 클라이언트 연결
    await manager.connect(websocket, symbol)
    start_symbol_thread(symbol)  # 데이터 캐시 쓰레드 시작
    try:
        last_sent = None
        while True:
            # 데이터가 바뀔 때만 push (또는 일정 주기마다)
            await asyncio.sleep(1)
            if symbol in data_cache:
                current = json.dumps(data_cache[symbol], sort_keys=True)
                if current != last_sent:
                    await websocket.send_text(current)
                    last_sent = current
    except WebSocketDisconnect:
        manager.disconnect(websocket, symbol)
    except Exception as e:
        manager.disconnect(websocket, symbol)

# -------- (END) --------
