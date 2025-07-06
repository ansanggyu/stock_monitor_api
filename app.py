#nohup uvicorn app:app --host 0.0.0.0 --port 8000 --reload > fastapi.log 2>&1 &

import yfinance as yf
import pandas as pd
import ta
import threading
import time
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from wcwidth import wcswidth

# ====== 분석 로직 함수 (기존 동일) ======

def get_data(symbol, period='5d', interval='1m'):
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        if all(val == symbol for val in df.columns.get_level_values(-1)):
            top_names = df.columns.get_level_values(0)
            colmap = {}
            for top, bot in zip(top_names, df.columns.get_level_values(-1)):
                if top.lower() in ['close','open','high','low','volume','adj close']:
                    colmap[(top, bot)] = top.title()
            df.columns = [colmap.get((t, b), t.title()) for t, b in zip(top_names, df.columns.get_level_values(-1))]
        else:
            df.columns = df.columns.get_level_values(-1)
    elif all(c == symbol for c in df.columns):
        if len(df.columns) == 5:
            df.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
        elif len(df.columns) == 6:
            df.columns = ['Close', 'High', 'Low', 'Open', 'Adj Close', 'Volume']
        else:
            return None
    if 'Close' not in df.columns:
        return None
    return df.dropna()

def calculate_indicators(df):
    close = df['Close']
    if isinstance(close, (pd.DataFrame, pd.Series)):
        close = close.squeeze()
    df['RSI'] = ta.momentum.RSIIndicator(close).rsi()
    macd = ta.trend.MACD(close)
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    bb = ta.volatility.BollingerBands(close)
    df['BBL'] = bb.bollinger_lband()
    df['BBH'] = bb.bollinger_hband()
    df['MA20'] = close.rolling(window=20, min_periods=1).mean()
    return df

def detect_signals(df):
    signal_dict = {}
    for i, (idx, row) in enumerate(df.iterrows()):
        signals = []
        def get_scalar(val):
            if isinstance(val, (pd.Series, list, tuple)):
                if len(val) == 1:
                    return val[0]
                elif hasattr(val, "item"):
                    return val.item()
                else:
                    return float(val[0])
            elif hasattr(val, "item"):
                return val.item()
            else:
                return val

        close_val = get_scalar(row['Close'])
        rsi_val = get_scalar(row['RSI'])
        macd_val = get_scalar(row['MACD'])
        macd_signal_val = get_scalar(row['MACD_signal'])
        bbl_val = get_scalar(row['BBL'])
        bbh_val = get_scalar(row['BBH'])
        ma20_val = get_scalar(row['MA20'])

        if i > 0 and not pd.isna(ma20_val):
            ma20_prev = get_scalar(df['MA20'].iloc[i-1])
            close_prev = get_scalar(df['Close'].iloc[i-1])
            EPS = 1e-8
            if close_prev <= ma20_prev + EPS and close_val > ma20_val + EPS:
                signals.append("🟢 20MA(20이평) 돌파(매수)")
            elif close_prev >= ma20_prev - EPS and close_val < ma20_val - EPS:
                signals.append("🔴 20MA(20이평) 이탈(매도)")

        if i > 0:
            macd_prev = get_scalar(df['MACD'].iloc[i-1])
            macd_signal_prev = get_scalar(df['MACD_signal'].iloc[i-1])
            macd_slope = macd_val - macd_prev
            if (macd_val > macd_signal_val) and (macd_prev <= macd_signal_prev):
                if macd_slope < 0.03:
                    phase = "관망 (약함)"
                elif macd_slope < 0.08:
                    phase = "매수 (보통)"
                else:
                    phase = "풀매수 (강함)"
                signals.append(f"🟢 골든크로스 - {phase} [{macd_slope:.4f}]")
            if (macd_val < macd_signal_val) and (macd_prev >= macd_signal_prev):
                if abs(macd_slope) < 0.03:
                    phase = "관망 (약함)"
                elif abs(macd_slope) < 0.08:
                    phase = "매도 (보통)"
                else:
                    phase = "풀매도 (강함)"
                signals.append(f"🔴 데드크로스 - {phase} [{macd_slope:.4f}]")

        if rsi_val is not None and not pd.isna(rsi_val):
            if rsi_val < 20:
                signals.append("🔵 극과매도(RSI<20)")
            elif rsi_val < 30:
                signals.append("🟦 과매도(RSI<30)")
            elif rsi_val > 80:
                signals.append("🟣 극과열(RSI>80)")
            elif rsi_val > 70:
                signals.append("🟨 과열(RSI>70)")
        if (
            macd_val is not None and not pd.isna(macd_val)
            and macd_signal_val is not None and not pd.isna(macd_signal_val)
        ):
            pass
        if (
            bbl_val is not None and not pd.isna(bbl_val)
            and bbh_val is not None and not pd.isna(bbh_val)
        ):
            if close_val < bbl_val:
                signals.append("🟫 볼밴 하단이탈")
            elif close_val > bbh_val:
                signals.append("🟧 볼밴 상단이탈")
        if i > 0:
            prev_min = float(df['Close'].iloc[:i].min())
            prev_max = float(df['Close'].iloc[:i].max())
            if close_val == prev_min:
                signals.append("🆕 신저가 갱신")
            if close_val == prev_max:
                signals.append("🆙 신고가 갱신")
        if signals:
            signal_dict[idx] = signals
    return signal_dict

def analyze_timeframes(symbol):
    intervals = [
        ('1m', '1분', '5d'),
        ('5m', '5분', '7d'),
        ('60m', '1시간', '30d'),
        ('1d', '일', '12mo'),
        ('1wk', '주', '7y'),
        ('1mo', '월', '30y'),
    ]
    result_lines = []
    for intv, name, period in intervals:
        df = get_data(symbol, period=period, interval=intv)
        if df is None or len(df) < 30:
            result_lines.append(f"📊 [{name}] 데이터 부족")
            continue
        df = calculate_indicators(df)
        latest = df.iloc[-1]
        cross_str = ""
        ma20_str = ""
        if len(df) > 1:
            prev = df.iloc[-2]
            macd = latest['MACD']
            signal = latest['MACD_signal']
            macd_prev = prev['MACD']
            signal_prev = prev['MACD_signal']
            EPS = 1e-5
            if (macd > signal + EPS) and (macd_prev <= signal_prev + EPS):
                cross_str = "🟢 골든크로스 발생!"
            elif (macd < signal - EPS) and (macd_prev >= signal_prev - EPS):
                cross_str = "🔴 데드크로스 발생!"

            if intv in ['1m', '5m', '60m', '1d']:
                close = latest['Close']
                close_prev = prev['Close']
                ma20 = latest['MA20']
                ma20_prev = prev['MA20']
                if (close_prev <= ma20_prev + EPS) and (close > ma20 + EPS):
                    ma20_str = "🟢 20MA 돌파(매수 시그널!)"
                elif (close_prev >= ma20_prev - EPS) and (close < ma20 - EPS):
                    ma20_str = "🔴 20MA 이탈(매도 시그널!)"
                elif close > ma20:
                    ma20_str = "🟩 20MA 위(강세 유지)"
                else:
                    ma20_str = "🟥 20MA 아래(약세 유지)"

        rsi = latest['RSI']
        price_now = latest['Close']
        macd = latest['MACD']
        signal = latest['MACD_signal']
        macd_slope = macd - df['MACD'].iloc[-2] if len(df) > 1 else 0
        bb_mid = (latest['BBH'] + latest['BBL']) / 2 if ('BBH' in df.columns and 'BBL' in df.columns) else None

        判 = []

        if rsi >= 80:
            判.append('🟣 극과열')
        elif rsi >= 70:
            判.append('🟨 과열')
        elif 50 <= rsi < 70:
            判.append('🟩 약세 조짐')
        elif 30 <= rsi < 50:
            判.append('🟦 약세 구간')
        elif rsi < 30:
            判.append('🔵 과매도')
        else:
            判.append(f'⚪ 정상({rsi:.1f})')

        try:
            if latest['Close'] > latest['BBH']:
                判.append('🔺 볼밴 상단 이탈')
            elif latest['Close'] < latest['BBL']:
                判.append('🔻 볼밴 하단 이탈')
            elif bb_mid is not None:
                if latest['Close'] > bb_mid:
                    判.append('🔸 볼밴 중앙선 위')
                else:
                    判.append('🔹 볼밴 중앙선 아래')
        except Exception:
            pass

        macd_signal_diff = abs(macd - signal)
        MACD_SIGNAL_THRESHOLD = 0.02
        if macd_signal_diff < MACD_SIGNAL_THRESHOLD:
            判.append('🔶 조정/관망 (신호 미약)')
        else:
            if macd > signal and macd > 0 and macd_slope > 0:
                判.append('📈 확실한 상승 추세')
            elif macd < signal and macd < 0 and macd_slope < 0:
                判.append('📉 확실한 하락 추세')
            else:
                判.append('🔶 조정/관망')

        판별 = ' + '.join(判) if 判 else '⚪ 관망 또는 판별 불가'
        detail_sig = " ".join([v for v in [cross_str, ma20_str] if v])
        if detail_sig:
            판별 = detail_sig + " + " + 판별

        result_lines.append(
            f"📊 [{name}] RSI: {rsi:.2f}, MACD: {macd:.4f}, Signal: {signal:.4f}, 20MA: {latest['MA20']:.4f}, 현재가: 💵 {price_now:.4f}\n→ 판별: {판별}"
        )
    return '\n'.join(result_lines)

def backtest_signals(df, signal_dict, holding_days=5):
    results = []
    for idx, signals in signal_dict.items():
        entry_idx = df.index.get_loc(idx)
        if entry_idx + holding_days >= len(df):
            continue
        entry_price = df.iloc[entry_idx]['Close']
        exit_idx = df.index[entry_idx + holding_days]
        exit_price = df.loc[exit_idx, 'Close']
        if isinstance(entry_price, (pd.Series, list, tuple)):
            entry_price = float(entry_price.squeeze())
        if isinstance(exit_price, (pd.Series, list, tuple)):
            exit_price = float(exit_price.squeeze())
        ret = (exit_price - entry_price) / entry_price
        results.append({
            'signal_time': idx,
            'signals': signals,
            'entry_price': entry_price,
            'exit_time': exit_idx,
            'exit_price': exit_price,
            'ret': ret
        })
    return pd.DataFrame(results)

def pad_display_width(s, width):
    real_width = wcswidth(str(s))
    pad_len = max(0, width - real_width)
    return str(s) + " " * pad_len

def make_signal_report(symbol, df, signal_dict, backtest_result):
    COL_DATE   = 12
    COL_SIG    = 42
    COL_ENTRY  = 12
    COL_EXIT_D = 12
    COL_EXIT   = 12
    COL_RET    = 9
    SEP        = 1
    TOTAL_W    = COL_DATE + COL_SIG + COL_ENTRY + COL_EXIT_D + COL_EXIT + COL_RET + SEP * 5

    latest_idx = df.index[-1]
    latest_row = df.iloc[-1]
    close_val = float(latest_row['Close'])

    lines = []
    lines.append("="*TOTAL_W)
    lines.append(f"[{symbol}] {latest_idx:%Y-%m-%d}  |  현재가: {close_val:.2f}")
    lines.append("-"*TOTAL_W)

    sigs = signal_dict.get(latest_idx, [])
    sig_icons = []
    for sig in sigs:
        if '골든크로스' in sig or '과매도' in sig:
            sig_icons.append("🟢" if "골든" in sig else "🟦" if "과매도" in sig else "■")
        elif '데드크로스' in sig or '과열' in sig:
            sig_icons.append("🔴" if "데드" in sig else "🟨" if "과열" in sig else "□")
        elif '볼밴 상단 이탈' in sig or '볼밴 상단이탈' in sig:
            sig_icons.append("🟧")
        else:
            sig_icons.append("■")
    sig_line = "◆ 최신 시그널 : "
    if sigs:
        sig_line += " ".join(f"{icon} {sig}" for icon, sig in zip(sig_icons, sigs))
    else:
        sig_line += "관망/신호 없음"
    lines.append(sig_line)
    lines.append("-"*TOTAL_W)

    header = (
        pad_display_width("일자", COL_DATE) + " " * SEP +
        pad_display_width("신호", COL_SIG) + " " * SEP +
        f"{'진입가':>{COL_ENTRY}}" + " " * SEP +
        f"{'청산일':>{COL_EXIT_D}}" + " " * SEP +
        f"{'청산가':>{COL_EXIT}}" + " " * SEP +
        f"{'수익률':>{COL_RET}}"
    )
    lines.append(header)

    if not backtest_result.empty:
        for _, row in backtest_result.tail(10).iterrows():
            sigstr = ', '.join(row['signals'])
            if wcswidth(sigstr) > COL_SIG:
                w, tmp = 0, ''
                for ch in sigstr:
                    w += wcswidth(ch)
                    if w >= COL_SIG-2: break
                    tmp += ch
                sigstr = tmp + '…'
            date_str = pad_display_width(f"{row['signal_time']:%Y-%m-%d}", COL_DATE)
            sig_disp = pad_display_width(sigstr, COL_SIG)
            entry_price = f"{row['entry_price']:{COL_ENTRY}.2f}".rjust(COL_ENTRY)
            exit_date = f"{row['exit_time']:%Y-%m-%d}".rjust(COL_EXIT_D)
            exit_price = f"{row['exit_price']:{COL_EXIT}.2f}".rjust(COL_EXIT)
            ret_str = f"{row['ret']*100:>{COL_RET-1}.2f}%"
            lines.append(
                date_str + " " * SEP +
                sig_disp + " " * SEP +
                entry_price + " " * SEP +
                exit_date + " " * SEP +
                exit_price + " " * SEP +
                ret_str
            )
    lines.append("-"*TOTAL_W)
    avg_str = f"{backtest_result['ret'].mean()*100:.2f}%" if not backtest_result.empty else "데이터 없음"
    lines.append(pad_display_width("신호평균수익률", TOTAL_W-13) + f": {avg_str}")
    lines.append("="*TOTAL_W)
    return "\n".join(lines)

# ====== 동적 캐싱, 에러 심볼 자동정리 스레드 ======

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
CACHE_INTERVAL = 1  # 1초마다

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
    max_error_count = 5  # 연속 5회 이상 실패시 종료
    while True:
        try:
            df = get_data(symbol)
            if not is_valid_data(df):
                error_count += 1
                print(f"[{symbol}] 데이터 없음/이상 ({error_count})")
                if error_count >= max_error_count:
                    print(f"[{symbol}] 5회 연속 실패. 캐싱 중지/심볼 제거")
                    watched_symbols.discard(symbol)
                    symbol_threads.pop(symbol, None)
                    data_cache.pop(symbol, None)
                    break
                time.sleep(3)
                continue
            # 정상 데이터면
            error_count = 0
            df = calculate_indicators(df)
            signal_dict = detect_signals(df)
            backtest_result = backtest_signals(df, signal_dict)
            result_text = make_signal_report(symbol, df, signal_dict, backtest_result)
            mtf_report = analyze_timeframes(symbol)
            data_cache[symbol] = {
                "symbol": symbol,
                "signal_report": result_text,
                "mtf_report": mtf_report,
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
        except Exception as e:
            print(f"[{symbol}] 캐시 갱신 예외: {e}")
            error_count += 1
            if error_count >= max_error_count:
                print(f"[{symbol}] 5회 연속 예외. 캐싱 중지/심볼 제거")
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

@app.get("/monitor")
def monitor(symbol: str = Query("TQQQ")):
    start_symbol_thread(symbol)
    if symbol in data_cache:
        return data_cache[symbol]
    else:
        return {"error": "아직 데이터 없음/잘못된 심볼일 수 있음", "symbol": symbol}
