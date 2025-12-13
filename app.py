import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Crash Hunter Strategy", layout="wide")
st.title("🐻 Crash Hunter Strategy Backtester")

# --- SIDEBAR INPUTS ---
st.sidebar.header("Strategy Settings")
symbol = st.sidebar.text_input("Ticker Symbol", "QQQ")
vix_symbol = st.sidebar.text_input("VIX Symbol", "^VIX")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-04-05"))
initial_capital = st.sidebar.number_input("Starting Cash ($)", 10000, 1000000, 100000)

st.sidebar.subheader("Entry Logic")
di_threshold = st.sidebar.slider("DI Spread Threshold (>)", 10, 50, 20)
vix_min = st.sidebar.slider("Min Absolute VIX (>)", 10, 50, 20)
vix_relative = st.sidebar.slider("VIX % of 20-Day High", 0.5, 1.0, 0.85)

st.sidebar.subheader("Risk Management")
risk_per_trade = st.sidebar.slider("Risk per Trade (%)", 1, 10, 2)
stop_loss_pct = st.sidebar.slider("Stop Loss Trail (%)", 1, 10, 5)

if st.sidebar.button("Run Backtest 🚀"):
    
    # --- 1. DATA LOADING ---
    with st.spinner(f"Downloading data for {symbol}..."):
        # Added auto_adjust=True to fix recent yfinance update warnings
        spy = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
        vix = yf.download(vix_symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)

        if isinstance(spy.columns, pd.MultiIndex): spy.columns = spy.columns.get_level_values(0)
        if isinstance(vix.columns, pd.MultiIndex): vix.columns = vix.columns.get_level_values(0)

        # Merge
        df = spy.copy()
        df['VIX'] = vix['Close']
        df = df.dropna()

    # --- 2. INDICATORS ---
    # Bollinger Bands
    window = 20
    df['SMA'] = df['Close'].rolling(window).mean()
    df['STD'] = df['Close'].rolling(window).std()
    df['Lower_BB2'] = df['SMA'] - (2 * df['STD'])

    # ADX / DMI
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(14).mean()

    df['UpMove'] = df['High'] - df['High'].shift(1)
    df['DownMove'] = df['Low'].shift(1) - df['Low']
    df['PlusDM'] = np.where((df['UpMove'] > df['DownMove']) & (df['UpMove'] > 0), df['UpMove'], 0)
    df['MinusDM'] = np.where((df['DownMove'] > df['UpMove']) & (df['DownMove'] > 0), df['DownMove'], 0)

    # Smooth
    plus_dm_series = pd.Series(df['PlusDM'].values, index=df.index)
    minus_dm_series = pd.Series(df['MinusDM'].values, index=df.index)
    
    df['PlusDI'] = 100 * (plus_dm_series.rolling(14).mean() / df['ATR'])
    df['MinusDI'] = 100 * (minus_dm_series.rolling(14).mean() / df['ATR'])
    
    # Helpers
    df['VIX_High_20'] = df['VIX'].rolling(20).max()
    df['Support_10'] = df['Low'].rolling(10).min().shift(1) 

    # --- 3. TRADING LOGIC ---
    cash = initial_capital
    shares = 0
    trade_log = []
    equity_curve = []
    
    current_month = -1
    monthly_start_equity = initial_capital
    locked_out = False

    for i in range(50, len(df)):
        date = df.index[i]
        row = df.iloc[i]
        price = row['Close']
        
        # Equity Tracking
        current_equity = cash + (shares * price)
        equity_curve.append({'Date': date, 'Equity': current_equity})

        # 6% Rule Check
        if date.month != current_month:
            current_month = date.month
            monthly_start_equity = current_equity
            locked_out = False
        
        if not locked_out and ((current_equity - monthly_start_equity) / monthly_start_equity < -0.06):
            locked_out = True
            if shares > 0:
                cash += shares * price
                shares = 0
                trade_log.append({'Date': date.date(), 'Type': 'SHARK BITE', 'Price': price, 'Shares': 0, 'PnL': 0})

        if locked_out: continue

        # Entry Logic
        if shares == 0:
            price_condition = price < row['Lower_BB2']
            di_spread = row['MinusDI'] - row['PlusDI']
            di_condition = (row['MinusDI'] > row['PlusDI']) and (di_spread > di_threshold)
            vix_condition = (row['VIX'] > vix_min) and (row['VIX'] >= (row['VIX_High_20'] * vix_relative))

            if price_condition and di_condition and vix_condition:
                # Sizing
                stop_level = row['Support_10']
                if stop_level >= price: stop_level = price * (1 - (stop_loss_pct/100))
                
                risk_per_share = price - stop_level
                risk_dollars = current_equity * (risk_per_trade/100)
                
                # Check for zero division
                if risk_per_share <= 0: risk_per_share = 0.01 
                
                size = int(risk_dollars / risk_per_share)
                if size * price > cash: size = int(cash / price)

                if size > 0:
                    shares = size
                    cash -= shares * price
                    trade_log.append({'Date': date.date(), 'Type': 'BUY', 'Price': price, 'Shares': shares, 'PnL': 0})

        # Exit Logic
        elif shares > 0:
            # Re-calc stop level 
            stop_line = row['Support_10']
            
            if row['Low'] < stop_line:
                cash += shares * stop_line
                trade_log.append({'Date': date.date(), 'Type': 'STOP LOSS', 'Price': stop_line, 'Shares': shares, 'PnL': -1})
                shares = 0
            
            elif price > df['SMA'][i]:
                cash += shares * price
                trade_log.append({'Date': date.date(), 'Type': 'TARGET HIT', 'Price': price, 'Shares': shares, 'PnL': 1})
                shares = 0

    # --- 4. DISPLAY RESULTS ---
    equity_df = pd.DataFrame(equity_curve).set_index('Date')
    final_val = equity_df['Equity'].iloc[-1]
    return_pct = ((final_val - initial_capital) / initial_capital) * 100
    
    # Top Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Final Portfolio Value", f"${final_val:,.2f}")
    col2.metric("Total Return", f"{return_pct:.2f}%")
    col3.metric("Total Trades", len([t for t in trade_log if t['Type'] == 'BUY']))

    # Charts
    st.subheader("Equity Curve")
    st.line_chart(equity_df)

    # Trade Log
    st.subheader("Trade Log")
    if trade_log:
        trades_df = pd.DataFrame(trade_log)
        st.dataframe(trades_df)
    else:
        st.warning("No trades found with these settings.")

else:
    st.info("👈 Adjust settings in the sidebar and click 'Run Backtest'")
