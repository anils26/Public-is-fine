"""
Price Action AI
---------------
Tabs: Single stock analysis + multi-stock Scanner. Timeframes: 5m to weekly.
Patterns: engulfing, hammer, shooting star, breakouts, trend pullbacks, double top/bottom,
head & shoulders (and inverse), bull/bear flags.

Enter a stock ticker -> the app:
  1. Downloads price history
  2. Detects price action patterns on EVERY candle in history
  3. Backtests each signal (entry, stop, target) and marks them on the chart
  4. Checks what pattern/setup is active NOW, shows its historical win rate
     and lists the reasons behind the trade idea
  5. (Optional) Asks Claude to write a plain-English summary

Install:  pip install streamlit yfinance plotly pandas numpy anthropic
Run:      streamlit run price_action_ai.py

Tickers: AAPL, TSLA, RELIANCE.NS (NSE), TCS.BO (BSE), BTC-USD, etc.
Educational tool only. Not financial advice.
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Price Action AI", layout="wide")

BAR_TD = {
    "5m": pd.Timedelta(minutes=5), "15m": pd.Timedelta(minutes=15),
    "30m": pd.Timedelta(minutes=30), "1h": pd.Timedelta(hours=1),
    "1d": pd.Timedelta(days=1), "1wk": pd.Timedelta(days=7),
}
# Yahoo Finance limits how far back intraday data goes
INTERVAL_PERIODS = {
    "1d": ["2y", "5y", "10y", "max"],
    "1wk": ["5y", "10y", "max"],
    "1h": ["6mo", "1y", "2y"],
    "30m": ["1mo", "60d"],
    "15m": ["1mo", "60d"],
    "5m": ["1mo", "60d"],
}

# --------------------------------------------------------------------------
# Data + indicators
# --------------------------------------------------------------------------


@st.cache_data(ttl=3600, show_spinner=False)
def load_data(ticker, period, interval):
    df = yf.download(ticker, period=period, interval=interval,
                     auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    return df[["open", "high", "low", "close", "volume"]].dropna()


def add_indicators(df):
    df = df.copy()
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    df["ema20"] = c.ewm(span=20, adjust=False).mean()
    df["sma50"] = c.rolling(50).mean()
    df["sma200"] = c.rolling(200).mean()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    d = c.diff()
    gain = d.clip(lower=0).rolling(14).mean()
    loss = (-d.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    df["vol_avg"] = v.rolling(20).mean()
    df["hh20"] = h.rolling(20).max().shift(1)
    df["ll20"] = l.rolling(20).min().shift(1)
    return df


# --------------------------------------------------------------------------
# Pattern detection (rule-based, evaluated on every candle)
# --------------------------------------------------------------------------

PATTERN_WHY = {
    "Bullish Engulfing": "A green candle fully engulfed the prior red candle after a pullback. "
                         "Buyers overwhelmed sellers in one session, a common reversal signal.",
    "Bearish Engulfing": "A red candle fully engulfed the prior green candle after a rally. "
                         "Sellers overwhelmed buyers, a common reversal signal.",
    "Hammer": "Long lower wick after a decline: price was pushed down but buyers rejected "
              "the lows and closed near the top of the range.",
    "Shooting Star": "Long upper wick after a rally: price was pushed up but sellers rejected "
                     "the highs and closed near the bottom of the range.",
    "Breakout (20-bar high)": "Close above the highest high of the last 20 bars on above-average "
                              "volume: buyers cleared resistance with participation.",
    "Breakdown (20-bar low)": "Close below the lowest low of the last 20 bars on above-average "
                              "volume: sellers broke support with participation.",
    "Trend Pullback Long": "In an uptrend (50 SMA > 200 SMA, price above 50 SMA), price dipped to the "
                           "20 EMA and closed back above it on a green candle: buying the dip.",
    "Trend Pullback Short": "In a downtrend (50 SMA < 200 SMA, price below 50 SMA), price rallied to the "
                            "20 EMA and closed back below it on a red candle: selling the rally.",
    "Double Top": "Price failed twice at about the same high, then closed below the support (neckline) "
                  "between the two peaks. Buyers could not push higher and sellers took control.",
    "Double Bottom": "Price held twice at about the same low, then closed above the resistance (neckline) "
                     "between the two troughs. Sellers could not push lower and buyers took control.",
    "Head & Shoulders": "A higher peak (head) between two lower, similar peaks (shoulders), followed by a "
                        "close below the neckline. Upward momentum has faded and the trend may be reversing.",
    "Inverse Head & Shoulders": "A lower trough (head) between two shallower, similar troughs (shoulders), "
                                "followed by a close above the neckline. Downward momentum has faded and "
                                "the trend may be reversing up.",
    "Bull Flag": "A sharp rally (the pole) was followed by a tight, shallow pause (the flag), then a close "
                 "above the flag's high. This often signals the prior up-move is resuming.",
    "Bear Flag": "A sharp drop (the pole) was followed by a tight, shallow bounce (the flag), then a close "
                 "below the flag's low. This often signals the prior down-move is resuming.",
}


def detect_patterns(df):
    o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    prev_o, prev_c = o.shift(1), c.shift(1)
    vol_ok = v > 1.2 * df["vol_avg"]

    down_before = c.shift(1) < c.shift(4)
    up_before = c.shift(1) > c.shift(4)

    sig = {}
    sig["Bullish Engulfing"] = (1, (prev_c < prev_o) & (c > o) & (c >= prev_o) & (o <= prev_c)
                                & (c.shift(1) < df["ema20"].shift(1)))
    sig["Bearish Engulfing"] = (-1, (prev_c > prev_o) & (c < o) & (o >= prev_c) & (c <= prev_o)
                                & (c.shift(1) > df["ema20"].shift(1)))
    sig["Hammer"] = (1, (lower >= 2 * body) & (upper <= 0.5 * body.clip(lower=1e-9))
                     & (body > 0.05 * rng) & down_before)
    sig["Shooting Star"] = (-1, (upper >= 2 * body) & (lower <= 0.5 * body.clip(lower=1e-9))
                            & (body > 0.05 * rng) & up_before)
    sig["Breakout (20-bar high)"] = (1, (c > df["hh20"]) & vol_ok)
    sig["Breakdown (20-bar low)"] = (-1, (c < df["ll20"]) & vol_ok)
    sig["Trend Pullback Long"] = (1, (df["sma50"] > df["sma200"]) & (c > df["sma50"])
                                  & (l <= df["ema20"]) & (c > df["ema20"]) & (c > o))
    sig["Trend Pullback Short"] = (-1, (df["sma50"] < df["sma200"]) & (c < df["sma50"])
                                   & (h >= df["ema20"]) & (c < df["ema20"]) & (c < o))
    # Bull / bear flags: strong pole, tight pause, breakout of the pause
    atr = df["atr"]
    flag_hi = h.rolling(5).max().shift(1)
    flag_lo = l.rolling(5).min().shift(1)
    tight = (flag_hi - flag_lo) <= 2.5 * atr
    pole_up = c.shift(6) - c.shift(16)
    pole_dn = c.shift(16) - c.shift(6)
    sig["Bull Flag"] = (1, tight & (pole_up >= 5 * atr) & (c > flag_hi)
                        & (flag_lo >= c.shift(6) - 0.5 * pole_up))
    sig["Bear Flag"] = (-1, tight & (pole_dn >= 5 * atr) & (c < flag_lo)
                        & (flag_hi <= c.shift(6) + 0.5 * pole_dn))

    out = {k: (d, m.fillna(False)) for k, (d, m) in sig.items()}
    out.update(detect_structure(df))
    return out


def detect_structure(df, k=5):
    """Double top/bottom and (inverse) head & shoulders from swing pivots.
    A pivot only becomes usable k bars after it forms, so there is no look-ahead."""
    n = len(df)
    h, l, c, atr = df["high"].values, df["low"].values, df["close"].values, df["atr"].values
    is_ph, is_pl = np.zeros(n, bool), np.zeros(n, bool)
    for i in range(k, n - k):
        if h[i] == h[i - k:i + k + 1].max():
            is_ph[i] = True
        if l[i] == l[i - k:i + k + 1].min():
            is_pl[i] = True

    names = ["Double Top", "Double Bottom", "Head & Shoulders", "Inverse Head & Shoulders"]
    m = {nm: np.zeros(n, bool) for nm in names}
    highs, lows, used = [], [], set()

    for i in range(1, n):
        p = i - k
        if p >= k:
            if is_ph[p]:
                highs.append((p, h[p]))
            if is_pl[p]:
                lows.append((p, l[p]))
        if np.isnan(atr[i]):
            continue
        a = atr[i]

        # Double top (short on neckline break)
        if len(highs) >= 2:
            (p1, h1), (p2, h2) = highs[-2], highs[-1]
            key = ("DT", p1, p2)
            if 8 <= p2 - p1 <= 80 and i - p2 <= 40 and abs(h1 - h2) <= max(0.02 * h2, a) \
                    and key not in used:
                neck = l[p1:p2 + 1].min()
                if c[i] < neck <= c[i - 1]:
                    m["Double Top"][i] = True
                    used.add(key)
        # Double bottom (long on neckline break)
        if len(lows) >= 2:
            (p1, l1), (p2, l2) = lows[-2], lows[-1]
            key = ("DB", p1, p2)
            if 8 <= p2 - p1 <= 80 and i - p2 <= 40 and abs(l1 - l2) <= max(0.02 * l2, a) \
                    and key not in used:
                neck = h[p1:p2 + 1].max()
                if c[i] > neck >= c[i - 1]:
                    m["Double Bottom"][i] = True
                    used.add(key)
        # Head & shoulders (short)
        if len(highs) >= 3:
            (p1, h1), (p2, h2), (p3, h3) = highs[-3:]
            key = ("HS", p1, p2, p3)
            if p3 - p1 <= 120 and i - p3 <= 40 and h2 - max(h1, h3) >= 0.5 * a \
                    and abs(h1 - h3) <= max(0.05 * h2, 1.5 * a) and key not in used:
                neck = (l[p1:p2 + 1].min() + l[p2:p3 + 1].min()) / 2
                if c[i] < neck <= c[i - 1]:
                    m["Head & Shoulders"][i] = True
                    used.add(key)
        # Inverse head & shoulders (long)
        if len(lows) >= 3:
            (p1, l1), (p2, l2), (p3, l3) = lows[-3:]
            key = ("IHS", p1, p2, p3)
            if p3 - p1 <= 120 and i - p3 <= 40 and min(l1, l3) - l2 >= 0.5 * a \
                    and abs(l1 - l3) <= max(0.05 * l3, 1.5 * a) and key not in used:
                neck = (h[p1:p2 + 1].max() + h[p2:p3 + 1].max()) / 2
                if c[i] > neck >= c[i - 1]:
                    m["Inverse Head & Shoulders"][i] = True
                    used.add(key)

    dirs = {"Double Top": -1, "Double Bottom": 1, "Head & Shoulders": -1,
            "Inverse Head & Shoulders": 1}
    return {nm: (dirs[nm], pd.Series(m[nm], index=df.index)) for nm in names}


# --------------------------------------------------------------------------
# Backtest
# --------------------------------------------------------------------------


def backtest(df, sigs, rr, atr_mult, max_hold):
    o, h, l, c, atr = (df[x].values for x in ["open", "high", "low", "close", "atr"])
    n = len(df)
    trades = []
    for name, (direction, mask) in sigs.items():
        last_exit = -1
        for i in np.where(mask.values)[0]:
            if i + 1 >= n or np.isnan(atr[i]) or i <= last_exit:
                continue
            entry = o[i + 1]
            risk = atr[i] * atr_mult
            if risk <= 0:
                continue
            stop = entry - direction * risk
            target = entry + direction * risk * rr
            outcome, exit_i, exit_p = "timeout", min(i + max_hold, n - 1), c[min(i + max_hold, n - 1)]
            for j in range(i + 1, min(i + 1 + max_hold, n)):
                hit_stop = l[j] <= stop if direction == 1 else h[j] >= stop
                hit_tgt = h[j] >= target if direction == 1 else l[j] <= target
                if hit_stop:  # stop checked first = conservative
                    outcome, exit_i, exit_p = "loss", j, stop
                    break
                if hit_tgt:
                    outcome, exit_i, exit_p = "win", j, target
                    break
            if outcome == "timeout" and i + max_hold >= n:
                outcome = "open"
            trades.append(dict(
                pattern=name, direction=direction,
                signal_date=df.index[i], entry_date=df.index[i + 1], entry=entry,
                exit_date=df.index[exit_i], exit=exit_p, stop=stop, target=target,
                outcome=outcome, r=direction * (exit_p - entry) / risk,
                bars=exit_i - i,
            ))
            last_exit = exit_i
    return pd.DataFrame(trades)


def pattern_stats(trades):
    if trades.empty:
        return pd.DataFrame()
    t = trades[trades["outcome"] != "open"]
    rows = []
    for name, x in t.groupby("pattern"):
        gains, losses = x.r[x.r > 0].sum(), -x.r[x.r < 0].sum()
        rows.append({
            "Pattern": name,
            "Trades": len(x),
            "Target hit %": round((x.outcome == "win").mean() * 100, 1),
            "Win rate % (profitable)": round((x.r > 0).mean() * 100, 1),
            "Avg R": round(x.r.mean(), 2),
            "Profit factor": round(gains / losses, 2) if losses > 0 else float("inf"),
            "Avg bars held": round(x.bars.mean(), 1),
        })
    return pd.DataFrame(rows).sort_values("Avg R", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Levels + context
# --------------------------------------------------------------------------


def pivots(df, k=5):
    hi, lo = df["high"].values, df["low"].values
    ph, pl = [], []
    for i in range(k, len(df) - k):
        if hi[i] == hi[i - k:i + k + 1].max():
            ph.append(hi[i])
        if lo[i] == lo[i - k:i + k + 1].min():
            pl.append(lo[i])
    return ph[-15:], pl[-15:]


def key_levels(df):
    price = df["close"].iloc[-1]
    ph, pl = pivots(df)
    above = [x for x in ph if x > price]
    below = [x for x in pl if x < price]
    res = min(above) if above else df["high"].tail(60).max()
    sup = max(below) if below else df["low"].tail(60).min()
    return sup, res


def context_points(df, sup, res):
    last = df.iloc[-1]
    pts = []
    if pd.notna(last["sma200"]):
        trend = "uptrend" if last["sma50"] > last["sma200"] else "downtrend"
        pts.append(f"Trend: {trend} (50 SMA {'above' if trend == 'uptrend' else 'below'} 200 SMA); "
                   f"price is {(last['close'] / last['sma200'] - 1) * 100:+.1f}% from the 200 SMA.")
    rsi = last["rsi"]
    if pd.notna(rsi):
        tag = "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral"
        pts.append(f"Momentum: RSI(14) = {rsi:.0f} ({tag}).")
    if pd.notna(last["vol_avg"]) and last["vol_avg"] > 0:
        pts.append(f"Volume: latest bar is {last['volume'] / last['vol_avg']:.1f}x the 20-bar average.")
    pts.append(f"Nearest support ~{sup:,.2f} ({(sup / last['close'] - 1) * 100:+.1f}%), "
               f"nearest resistance ~{res:,.2f} ({(res / last['close'] - 1) * 100:+.1f}%).")
    return pts


# --------------------------------------------------------------------------
# Chart
# --------------------------------------------------------------------------


def build_chart(df, trades, active, sup, res, show_lines, max_hold, interval, ticker):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.78, 0.22],
                        vertical_spacing=0.02)
    fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"], low=df["low"],
                                 close=df["close"], name="Price",
                                 increasing_line_color="#26a69a", decreasing_line_color="#ef5350"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["sma50"], name="SMA 50",
                             line=dict(color="#f9a825", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["sma200"], name="SMA 200",
                             line=dict(color="#7e57c2", width=1)), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df["volume"], name="Volume",
                         marker_color="rgba(120,120,120,0.5)"), row=2, col=1)

    colors = {"win": "#00c853", "loss": "#ff1744", "timeout": "#9e9e9e", "open": "#2979ff"}
    if not trades.empty:
        for outcome, col in colors.items():
            t = trades[trades["outcome"] == outcome]
            if t.empty:
                continue
            for direction, sym in [(1, "triangle-up"), (-1, "triangle-down")]:
                td = t[t["direction"] == direction]
                if td.empty:
                    continue
                fig.add_trace(go.Scatter(
                    x=td["entry_date"], y=td["entry"], mode="markers",
                    name=f"{'Long' if direction == 1 else 'Short'} - {outcome}",
                    marker=dict(symbol=sym, size=10, color=col, line=dict(width=1, color="white")),
                    text=[f"{r.pattern}<br>{'LONG' if r.direction == 1 else 'SHORT'} -> {r.outcome.upper()}"
                          f"<br>Entry {r.entry:,.2f} | Stop {r.stop:,.2f} | Target {r.target:,.2f}"
                          f"<br>Result {r.r:+.2f}R in {r.bars} bars" for r in td.itertuples()],
                    hoverinfo="text+x"), row=1, col=1)
                if show_lines:
                    xs, ys = [], []
                    for r in td.itertuples():
                        xs += [r.entry_date, r.exit_date, None]
                        ys += [r.entry, r.exit, None]
                    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", showlegend=False,
                                             line=dict(color=col, width=1, dash="dot"),
                                             hoverinfo="skip"), row=1, col=1)

    # Support / resistance
    for y, name, col in [(sup, "Support", "#00c853"), (res, "Resistance", "#ff1744")]:
        fig.add_hline(y=y, line=dict(color=col, width=1, dash="dash"),
                      annotation_text=f"{name} {y:,.2f}", row=1, col=1)

    # Projection for active setup(s)
    pad = BAR_TD[interval] * max_hold * (1.4 if interval == "1d" else 1)
    last_dt = df.index[-1]
    for a in active:
        for y, nm, col in [(a["entry"], "Entry", "#2979ff"), (a["stop"], "Stop", "#ff1744"),
                           (a["target"], "Target", "#00c853")]:
            fig.add_trace(go.Scatter(x=[last_dt, last_dt + pad], y=[y, y], mode="lines+text",
                                     text=["", f"{nm} {y:,.2f}"], textposition="top right",
                                     line=dict(color=col, width=2), showlegend=False,
                                     hoverinfo="skip"), row=1, col=1)

    right = last_dt + pad
    left = df.index[-300] if len(df) > 300 else df.index[0]
    fig.update_xaxes(range=[left, right], rangeslider_visible=False)
    if interval != "1wk":
        breaks = [dict(bounds=["sat", "mon"])]
        if interval != "1d":
            hrs = sorted(set(df.index.hour))
            if len(hrs) < 24 and hrs[-1] + 1 <= 24:  # hide overnight gap (not for 24h markets)
                breaks.append(dict(bounds=[hrs[-1] + 1, hrs[0]], pattern="hour"))
        fig.update_xaxes(rangebreaks=breaks)
    fig.update_layout(title=f"{ticker}: historical price action trades + current setup",
                      height=720, template="plotly_dark", margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", y=1.04))
    return fig


# --------------------------------------------------------------------------
# Current setup
# --------------------------------------------------------------------------


def current_setups(df, sigs, stats, rr, atr_mult):
    last = df.iloc[-1]
    risk = last["atr"] * atr_mult
    out = []
    for name, (direction, mask) in sigs.items():
        if bool(mask.iloc[-1]):
            row = stats[stats["Pattern"] == name] if not stats.empty else pd.DataFrame()
            out.append(dict(
                pattern=name, direction=direction, entry=last["close"],
                stop=last["close"] - direction * risk,
                target=last["close"] + direction * risk * rr,
                stats=row.iloc[0].to_dict() if not row.empty else None,
            ))
    return out


def claude_summary(prompt):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return "Set the ANTHROPIC_API_KEY environment variable to enable AI summaries."
    import anthropic
    client = anthropic.Anthropic(api_key=key)
    msg = client.messages.create(
        model="claude-sonnet-5-5", max_tokens=700,
        messages=[{"role": "user", "content": prompt}])
    return "".join(b.text for b in msg.content if b.type == "text")


# --------------------------------------------------------------------------
# Analysis pipeline (shared by single-stock view and scanner)
# --------------------------------------------------------------------------


def analyze(ticker, period, interval, chosen, rr, atr_mult, max_hold):
    raw = load_data(ticker, period, interval)
    if raw.empty or len(raw) < 250:
        return None
    df = add_indicators(raw)
    sigs = {k: v for k, v in detect_patterns(df).items() if k in chosen}
    trades = backtest(df, sigs, rr, atr_mult, max_hold)
    stats = pattern_stats(trades)
    sup, res = key_levels(df)
    active = current_setups(df, sigs, stats, rr, atr_mult)
    return dict(df=df, sigs=sigs, trades=trades, stats=stats, sup=sup, res=res, active=active)


def render_single(ticker, A, interval, rr, max_hold, show_lines):
    df, trades, stats, sup, res, active = (A[k] for k in
                                           ["df", "trades", "stats", "sup", "res", "active"])
    st.plotly_chart(build_chart(df, trades, active, sup, res, show_lines, max_hold, interval, ticker),
                    use_container_width=True)
    st.caption("Markers: green = target hit, red = stopped out, grey = timed out, blue = still open. "
               "Triangle up = long, down = short. Hover for trade details.")

    st.header("Current setup")
    for p in context_points(df, sup, res):
        st.write("- " + p)

    if active:
        for a in active:
            side = "LONG" if a["direction"] == 1 else "SHORT"
            st.subheader(f"{a['pattern']} ({side}) is active on the latest candle")
            c1, c2, c3 = st.columns(3)
            c1.metric("Entry (approx)", f"{a['entry']:,.2f}")
            c2.metric("Stop", f"{a['stop']:,.2f}")
            c3.metric("Target", f"{a['target']:,.2f}")
            st.write("**Why this trade:** " + PATTERN_WHY[a["pattern"]])
            s = a["stats"]
            if s:
                st.write(f"**History of this pattern on {ticker}:** {int(s['Trades'])} trades, "
                         f"target hit {s['Target hit %']}%, profitable {s['Win rate % (profitable)']}%, "
                         f"average {s['Avg R']:+.2f}R, profit factor {s['Profit factor']}.")
                if s["Trades"] < 30:
                    st.warning("Small sample (<30 trades). Treat these statistics as weak evidence.")
                if s["Avg R"] <= 0:
                    st.warning("This pattern has had zero or negative expectancy on this stock "
                               "with these rules.")
            else:
                st.warning("No completed historical trades for this pattern on this stock.")
    else:
        st.info("No pattern triggered on the latest candle. Watch these triggers instead:")
        st.write(f"- Breakout trigger: close above {df['hh20'].iloc[-1]:,.2f} (20-bar high) on strong volume.")
        st.write(f"- Breakdown trigger: close below {df['ll20'].iloc[-1]:,.2f} (20-bar low) on strong volume.")
        st.write(f"- Support {sup:,.2f} / resistance {res:,.2f} are the nearest reaction zones.")

    st.header("Historical performance by pattern")
    if stats.empty:
        st.write("No completed trades found.")
    else:
        st.dataframe(stats, use_container_width=True, hide_index=True)
        st.caption("R = multiple of the initial risk. Backtest assumes entry at next open, stop checked "
                   "before target within a bar, no fees or slippage, one trade at a time per pattern.")
        with st.expander("All historical trades"):
            show = trades.copy()
            show["direction"] = show["direction"].map({1: "Long", -1: "Short"})
            st.dataframe(show.sort_values("entry_date", ascending=False),
                         use_container_width=True, hide_index=True)

    st.header("AI summary")
    if st.button("Ask Claude to explain this chart"):
        prompt = (
            f"You are a price action analyst. Summarize this stock chart for a trader in plain English "
            f"(max 250 words). Be balanced, mention risks, and do not give financial advice.\n\n"
            f"Ticker: {ticker} ({interval})\nLast close: {df['close'].iloc[-1]:,.2f}\n"
            f"Context:\n- " + "\n- ".join(context_points(df, sup, res)) + "\n\n"
            f"Active setups: {active or 'none'}\n\n"
            f"Historical pattern stats:\n{stats.to_string(index=False) if not stats.empty else 'none'}"
        )
        with st.spinner("Thinking..."):
            st.write(claude_summary(prompt))


def run_scan(tickers, period, interval, chosen, rr, atr_mult, max_hold):
    rows, skipped, quiet = [], [], []
    bar, status = st.progress(0.0), st.empty()
    for n, t in enumerate(tickers):
        status.write(f"Scanning {t} ({n + 1}/{len(tickers)})...")
        bar.progress((n + 1) / len(tickers))
        try:
            A = analyze(t, period, interval, chosen, rr, atr_mult, max_hold)
        except Exception:
            A = None
        if A is None:
            skipped.append(t)
            continue
        if not A["active"]:
            quiet.append(t)
        for a in A["active"]:
            s = a["stats"] or {}
            rows.append({
                "Ticker": t, "Pattern": a["pattern"],
                "Side": "LONG" if a["direction"] == 1 else "SHORT",
                "Entry": round(a["entry"], 2), "Stop": round(a["stop"], 2),
                "Target": round(a["target"], 2),
                "Trades": s.get("Trades", 0),
                "Target hit %": s.get("Target hit %", np.nan),
                "Win rate %": s.get("Win rate % (profitable)", np.nan),
                "Avg R": s.get("Avg R", np.nan),
                "Profit factor": s.get("Profit factor", np.nan),
            })
    bar.empty()
    status.empty()
    return pd.DataFrame(rows), skipped, quiet


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

st.title("Price Action AI")
st.caption("Enter a stock, see every historical pattern trade, and what the current chart is setting up. "
           "Educational only, not financial advice.")

with st.sidebar:
    interval = st.selectbox("Timeframe", list(INTERVAL_PERIODS.keys()), index=0)
    periods = INTERVAL_PERIODS[interval]
    period = st.selectbox("History", periods, index=1 if interval == "1d" else 0)
    if interval not in ("1d", "1wk"):
        st.caption("Intraday history is limited by Yahoo Finance (5m-30m: about 60 days, 1h: about 2 years).")
    st.markdown("**Trade rules (backtest)**")
    rr = st.slider("Reward : Risk", 1.0, 4.0, 2.0, 0.5)
    atr_mult = st.slider("Stop distance (x ATR)", 0.5, 3.0, 1.5, 0.25)
    max_hold = st.slider("Max bars to hold", 5, 60, 20, 5)
    show_lines = st.checkbox("Draw entry -> exit lines", value=False)
    all_patterns = list(PATTERN_WHY.keys())
    chosen = st.multiselect("Patterns", all_patterns, default=all_patterns)

tab_single, tab_scan = st.tabs(["Single stock", "Scanner"])

with tab_single:
    ticker = st.text_input("Stock ticker", "AAPL").upper().strip()
    if ticker:
        with st.spinner(f"Analyzing {ticker}..."):
            A = analyze(ticker, period, interval, chosen, rr, atr_mult, max_hold)
        if A is None:
            st.error("Not enough data (need 250+ candles). Check the ticker (e.g. AAPL, RELIANCE.NS) "
                     "or choose a longer history.")
        else:
            render_single(ticker, A, interval, rr, max_hold, show_lines)

with tab_scan:
    st.write("Scan many stocks at once. The scanner lists every stock with a pattern active on its "
             "latest candle, ranked by that pattern's historical average R on that stock.")
    default_list = "AAPL, MSFT, NVDA, TSLA, AMZN, GOOGL, META, RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS"
    text = st.text_area("Tickers (comma or new line separated, max 60)", default_list, height=100)
    c1, c2 = st.columns(2)
    min_trades = c1.number_input("Minimum historical trades", 0, 200, 10)
    only_pos = c2.checkbox("Only patterns with positive average R", value=True)

    if st.button("Run scan"):
        tickers = list(dict.fromkeys(t.strip().upper() for t in text.replace("\n", ",").split(",")
                                     if t.strip()))[:60]
        if not tickers:
            st.warning("Enter at least one ticker.")
        else:
            res_df, skipped, quiet = run_scan(tickers, period, interval, chosen, rr, atr_mult, max_hold)
            if not res_df.empty:
                shown = res_df[res_df["Trades"] >= min_trades]
                if only_pos:
                    shown = shown[shown["Avg R"] > 0]
                shown = shown.sort_values("Avg R", ascending=False, na_position="last")
            else:
                shown = res_df
            if shown.empty:
                st.info(f"No setups passed the filters across {len(tickers) - len(skipped)} stocks scanned. "
                        "Try lowering the minimum trades or unticking the positive-R filter.")
            else:
                st.success(f"{len(shown)} setup(s) found.")
                st.dataframe(shown, use_container_width=True, hide_index=True)
                st.download_button("Download CSV", shown.to_csv(index=False), "scan_results.csv", "text/csv")
                st.caption("Open a ticker in the Single stock tab to see its chart and every historical trade.")
            if quiet:
                st.caption("No active pattern: " + ", ".join(quiet))
            if skipped:
                st.caption("Skipped (no data or too short): " + ", ".join(skipped))
