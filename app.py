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

# ==== 데이터 유틸 ====
def fix_df_columns(df, symbol):
    if isinstance(df.columns, pd.MultiIndex):
        # 멀티인덱스면 심볼 하위레벨만 남김
        df = df.xs(symbol, axis=1, level=-1)
    # 칼럼 이름 정제
    rename_dict = {c: c.title() for c in df.columns}
    df = df.rename(columns=rename_dict)
    return df

def get_data(symbol: str, period="7d", interval="1m") -> pd.DataFrame:
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(symbol, axis=1, level=-1)
        # 칼럼 표준화
        df.columns = [c.title() for c in df.columns]
        df = df.dropna()
    except Exception as e:
        print(f"[get_data] Error for {symbol}: {e}")
        df = pd.DataFrame()
    return df

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Close" not in df.columns:
        return df
    close = df["Close"]
    df["Rsi"] = ta.momentum.RSIIndicator(close, window=14).rsi()
    macd = ta.trend.MACD(close)
    df["Macd"] = macd.macd()
    df["Macd_signal"] = macd.macd_signal()
    bb = ta.volatility.BollingerBands(close)
    df["Bb_upper"] = bb.bollinger_hband()
    df["Bb_lower"] = bb.bollinger_lband()
    df["Ma20"] = close.rolling(window=20, min_periods=1).mean()
    return df

def detect_signals(df: pd.DataFrame) -> Dict[str, List[str]]:
    result = {}
    for idx, row in df.iterrows():
        sigs = []
        # RSI
        rsi = row.get("Rsi", 50)
        if rsi > 80:
            sigs.append("🟣 극과열(RSI>80)")
        elif rsi > 70:
            sigs.append("🟨 과열(RSI>70)")
        elif rsi < 20:
            sigs.append("🔵 극과매도(RSI<20)")
        elif rsi < 30:
            sigs.append("🟦 과매도(RSI<30)")
        # MACD
        if row.get("Macd", 0) > row.get("Macd_signal", 0):
            sigs.append("🟢 MACD 골든크로스")
        elif row.get("Macd", 0) < row.get("Macd_signal", 0):
            sigs.append("🔴 MACD 데드크로스")
        # MA20 돌파/이탈
        close = row.get("Close", 0)
        ma20 = row.get("Ma20", 0)
        if pd.notna(ma20):
            if close > ma20:
                sigs.append("🟩 20MA 위(강세)")
            elif close < ma20:
                sigs.append("🟥 20MA 아래(약세)")
        # 볼밴 상하단
        if close > row.get("Bb_upper", 0):
            sigs.append("🟧 볼밴 상단이탈")
        if close < row.get("Bb_lower", 0):
            sigs.append("🟫 볼밴 하단이탈")
        if sigs:
            result[str(idx)] = sigs
    return result

def backtest_signals(df: pd.DataFrame, signals: Dict[str, List[str]], holding=5) -> pd.DataFrame:
    result = []
    df = df.copy()
    for idx in signals.keys():
        try:
            i = df.index.get_loc(pd.Timestamp(idx))
        except Exception:
            try:
                i = df.index.get_loc(idx)
            except Exception:
                continue
        entry = df.iloc[i]
        entry_price = entry["Close"]
        entry_time = idx
        signal_list = signals[idx]
        exit_i = i+holding if i+holding < len(df) else len(df)-1
        exit = df.iloc[exit_i]
        exit_price = exit["Close"]
        exit_time = str(exit.name)
        ret = (exit_price - entry_price) / entry_price if entry_price else 0
        result.append({
            "signal_time": entry_time,
            "signals": signal_list,
            "entry_price": float(entry_price),
            "exit_time": exit_time,
            "exit_price": float(exit_price),
            "ret": float(ret)
        })
    return pd.DataFrame(result)

def analyze_timeframes(symbol: str) -> List[str]:
    intervals = [
        ("1m", "1분", "5d"),
        ("5m", "5분", "7d"),
        ("60m", "1시간", "30d"),
        ("1d", "일", "12mo"),
        ("1wk", "주", "7y"),
        ("1mo", "월", "30y"),
    ]
    results = []
    for intv, name, period in intervals:
        try:
            df = get_data(symbol, period=period, interval=intv)
            if df.empty or len(df) < 30 or "Close" not in df.columns:
                results.append(f"📊 [{name}] 데이터 부족")
                continue
            df = calculate_indicators(df)
            last = df.iloc[-1]
            macd = last["Macd"]
            macd_signal = last["Macd_signal"]
            rsi = last["Rsi"]
            close = last["Close"]
            ma20 = last["Ma20"] if "Ma20" in last else None
            bb_upper = last["Bb_upper"] if "Bb_upper" in last else None
            bb_lower = last["Bb_lower"] if "Bb_lower" in last else None
            判 = []
            if rsi >= 80:
                判.append('🟣 극과열')
            elif rsi >= 70:
                判.append('🟨 과열')
            elif rsi < 30:
                判.append('🔵 과매도')
            if macd > macd_signal:
                判.append("🟢 MACD 골든")
            elif macd < macd_signal:
                判.append("🔴 MACD 데드")
            if ma20 is not None and close > ma20:
                判.append("🟩 20MA 위")
            elif ma20 is not None and close < ma20:
                判.append("🟥 20MA 아래")
            if bb_upper is not None and close > bb_upper:
                判.append("🟧 볼밴 상단이탈")
            if bb_lower is not None and close < bb_lower:
                判.append("🟫 볼밴 하단이탈")
            res = " + ".join(判) if 判 else "신호 없음"
            results.append(
                f"📊 [{name}] RSI: {rsi:.2f}, MACD: {macd:.4f}, Signal: {macd_signal:.4f}, 현재가: 💵 {close:.2f}\n→ 판별: {res}"
            )
        except Exception as e:
            results.append(f"📊 [{name}] 에러: {e}")
    return results

def is_valid_data(df):
    return (
        df is not None and not df.empty
        and "Close" in df.columns
        and df['Close'].notnull().any()
        and (df['Close'] > 0).any()
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
            df = get_data(symbol, period="7d", interval="1m")
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
            if current_cache != last_cache:
                data_cache[symbol] = current_cache
                last_cache = current_cache
        except Exception as e:
            print(f"[update_cache] {symbol}: 예외 발생: {e}")
            error_count += 1
        if error_count >= max_error_count:
            watched_symbols.discard(symbol)
            symbol_threads.pop(symbol, None)
            data_cache.pop(symbol, None)
            print(f"[update_cache] {symbol}: 에러 카운트 초과, 감시 중단")
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
