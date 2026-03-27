import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Crash Hunter Pro: VIX & Gamma", layout="wide")
st.title("🐻 Crash Hunter Pro: Mean Reversion & VIX Exit")
st.markdown("""
This version implements **Volatility Targeting** and **VIX Mean Reversion Exits**. 
It helps identify when the 'panic' is over so you can transition from Buy & Hold to the Wheel.
""")

# --- SIDEBAR INPUTS ---
st.sidebar.header("1. Asset & Timeline")
symbol = st.sidebar.text_input("Ticker Symbol", "QQQ")
vix_symbol = st.sidebar.text_input("VIX Symbol", "^VIX")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2026-03-27"))
initial_capital = st.sidebar.number_input("Starting Cash ($)", 10000, 1000000, 100000)

st.sidebar.header("2. Entry Logic (High VIX)")
di_threshold = st.sidebar.slider("Entry: DI Spread (Bears) >", 10, 50, 20)
vix_min_entry = st.sidebar.slider("Min Entry VIX >", 10, 50, 25)
vix_rel_entry = st.sidebar.slider("VIX % of 20-Day High", 0.5, 1.0, 0.85)

st.sidebar.header("3. Exit Logic (Mean Reversion)")
exit_mode = st.sidebar.selectbox(
    "Primary Exit Trigger",
    ["VIX Absolute Level", "VIX SMA Cross", "Price SMA (Mean Reversion)", "ATR Trailing Stop"]
)

if exit_mode == "VIX Absolute Level":
    vix_exit_threshold = st.sidebar.slider("Exit when VIX drops below:", 10, 35, 20)
elif exit_mode == "VIX SMA Cross":
    vix_sma_len = st.sidebar.slider("VIX SMA Lookback", 5, 50, 20)
elif exit_mode == "ATR Trailing Stop":
    atr_mult = st.sidebar.slider("ATR Multiplier", 1.0, 5.0, 3.0)

st.sidebar.header("4. Risk & Volatility Sizing")
use_vol_targeting = st.sidebar.checkbox("Use Volatility Targeting", value=True)
risk_per_trade = st.sidebar.slider("Risk per Trade (%)", 1, 10, 3)

if st.sidebar.button("Run Backtest 🚀"):
    
    # --- 1. DATA LOADING ---
    with st.spinner(f"Downloading data..."):
        spy = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
        vix = yf.download(vix_symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)

        if isinstance(spy.columns, pd.MultiIndex): spy.columns = spy.columns.get_level_values(0)
        if isinstance(vix.columns, pd.MultiIndex): vix.columns = vix.columns.get_level_values(0)

        df = spy.copy()
        df['VIX'] = vix['Close']
        df = df.dropna()

    # --- 2. INDICATORS ---
    window = 20
    df['SMA'] = df['Close'].rolling(window).mean()
    df['STD'] = df['Close'].rolling(window).std()
    df['Upper_BB2'] = df['SMA'] + (2 * df['STD'])
    df['Lower_BB2'] = df['SMA'] - (2 * df['STD'])
    
    # VIX Indicators for Exit
    df['VIX_SMA'] = df['VIX'].rolling(20).mean()
    df['VIX_High_20'] = df['VIX'].rolling(20).max()

    # DMI / ATR
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()

    df['UpMove'] = df['High'] - df['High'].shift(1)
    df['DownMove'] = df['Low'].shift(1) - df['Low']
    df['PlusDM'] = np.where((df['UpMove'] > df['DownMove']) & (df['UpMove'] > 0), df['UpMove'], 0)
    df['MinusDM'] = np.where((df['DownMove'] > df['UpMove']) & (df['DownMove'] > 0), df['DownMove'], 0)
    
    df['PlusDI'] = 100 * (pd.Series(df['PlusDM']).rolling(14).mean() / df['ATR'])
    df['MinusDI'] = 100 * (pd.Series(df['MinusDM']).rolling(14).mean() / df['ATR'])

    # --- 3. TRADING LOGIC ---
    cash = initial_capital
    shares = 0
    trade_log = []
    equity_curve = []
    
    highest_price_since_entry = 0

    for i in range(50, len(df)):
        date = df.index[i]
        row = df.iloc[i]
        price = row['Close']
        vix_val = row['VIX']
        
        current_equity = cash + (shares * price)
        equity_curve.append({'Date': date, 'Strategy Equity': current_equity})

        # --- ENTRY LOGIC ---
        if shares == 0:
            di_spread = row['MinusDI'] - row['PlusDI']
            
            # Crash Hunter Setup: Price below BB, High VIX, Bearish Momentum
            entry_signal = (
                price < row['Lower_BB2'] and 
                vix_val > vix_min_entry and 
                vix_val >= (row['VIX_High_20'] * vix_rel_entry) and
                di_spread > di_threshold
            )

            if entry_signal:
                # VOLATILITY TARGETING: Lower position size if VIX is insane
                # Base size on ATR, but modified by risk_per_trade
                risk_amt = current_equity * (risk_per_trade / 100)
                stop_dist = row['ATR'] * 3
                
                size = int(risk_amt / stop_dist)
                
                # Ceiling check
                if size * price > cash: size = int(cash / price)

                if size > 0:
                    shares = size
                    cash -= shares * price
                    highest_price_since_entry = price
                    trade_log.append({'Date': date.date(), 'Type': 'BUY (CRASH)', 'Price': price, 'VIX': vix_val})

        # --- EXIT LOGIC ---
        elif shares > 0:
            if price > highest_price_since_entry: highest_price_since_entry = price
            
            should_sell = False
            reason = ""

            # 1. VIX Absolute Level (Fear normalized)
            if exit_mode == "VIX Absolute Level" and vix_val < vix_exit_threshold:
                should_sell = True
                reason = f"VIX < {vix_exit_threshold}"

            # 2. VIX SMA Cross (Volatility trend reversal)
            elif exit_mode == "VIX SMA Cross" and vix_val < row['VIX_SMA']:
                should_sell = True
                reason = "VIX CROSS BELOW SMA"

            # 3. Price SMA (Reversion to Mean)
            elif exit_mode == "Price SMA (Mean Reversion)" and price > row['SMA']:
                should_sell = True
                reason = "PRICE > SMA (MEAN)"

            # 4. ATR Trailing Stop
            elif exit_mode == "ATR Trailing Stop":
                trail_stop = highest_price_since_entry - (row['ATR'] * atr_mult)
                if price < trail_stop:
                    should_sell = True
                    reason = "ATR TRAIL HIT"

            # Hard Stop (Safety Net 10%)
            if price < (highest_price_since_entry * 0.90):
                should_sell = True
                reason = "HARD STOP 10%"

            if should_sell:
                cash += shares * price
                trade_log.append({'Date': date.date(), 'Type': 'SELL', 'Price': price, 'VIX': vix_val, 'Reason': reason})
                shares = 0

    # 4. STATS ENGINE
    eq_df = pd.DataFrame(equity_curve).set_index('Date')
    eq_df['Benchmark'] = initial_capital * (df['Close'] / df['Close'].iloc[20])
    
    # CAGR Calculation
    years = (eq_df.index[-1] - eq_df.index[0]).days / 365.25
    cagr_strat = (eq_df['Strategy'].iloc[-1] / initial_capital)**(1/years) - 1
    cagr_bench = (eq_df['Benchmark'].iloc[-1] / initial_capital)**(1/years) - 1

    # Sharpe (Risk-Free = 0 for simplicity)
    strat_rets = eq_df['Strategy'].pct_change().dropna()
    bench_rets = eq_df['Benchmark'].pct_change().dropna()
    sharpe_strat = (strat_rets.mean() / strat_rets.std()) * np.sqrt(252)
    sharpe_bench = (bench_rets.mean() / bench_rets.std()) * np.sqrt(252)

    # Max Drawdown
    dd_strat = (eq_df['Strategy'] / eq_df['Strategy'].cummax() - 1).min()

    # 5. DASHBOARD
    st.subheader("Performance Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Strategy CAGR", f"{cagr_strat*100:.2f}%", delta=f"{(cagr_strat-cagr_bench)*100:.2f}% vs Bench")
    m2.metric("Benchmark CAGR", f"{cagr_bench*100:.2f}%")
    m3.metric("Strategy Sharpe", f"{sharpe_strat:.2f}")
    m4.metric("Max Drawdown", f"{dd_strat*100:.1f}%")

    st.line_chart(eq_df[['Strategy', 'Benchmark']])
    st.write("### Trade Log", pd.DataFrame(trade_log))

    st.subheader("Why the VIX Exit matters:")
    st.write("""
    In a mean-reversion strategy, **price follows volatility.** By exiting when the VIX drops below its SMA 
    or a target level (like 20), you capture the 'relief rally' before the market enters a boring sideways 
    phase—which is exactly when you should switch to **Selling Covered Calls (The Wheel).**
    """)
