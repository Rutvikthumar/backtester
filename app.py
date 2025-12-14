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
di_threshold = st.sidebar.slider("Entry: DI Spread (Bears) >", 10, 50, 20)
vix_min = st.sidebar.slider("Min Absolute VIX >", 10, 50, 20)
vix_relative = st.sidebar.slider("VIX % of 20-Day High", 0.5, 1.0, 0.85)

st.sidebar.subheader("Exit Logic")
exit_strategy = st.sidebar.selectbox(
    "Take Profit Strategy",
    ("Middle Band (SMA)", "Upper Band (2-Sigma)", "Bullish DI Reversal (DI+ - DI- > 20)")
)

st.sidebar.subheader("Risk Management")
risk_per_trade = st.sidebar.slider("Risk per Trade (%)", 1, 10, 2)
stop_loss_pct = st.sidebar.slider("Stop Loss Trail (%)", 1, 10, 5)

if st.sidebar.button("Run Backtest 🚀"):
    
    # --- 1. DATA LOADING ---
    with st.spinner(f"Downloading data for {symbol}..."):
        spy = yf.download(symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)
        vix = yf.download(vix_symbol, start=start_date, end=end_date, progress=False, auto_adjust=True)

        if isinstance(spy.columns, pd.MultiIndex): spy.columns = spy.columns.get_level_values(0)
        if isinstance(vix.columns, pd.MultiIndex): vix.columns = vix.columns.get_level_values(0)

        # Merge
        df = spy.copy()
        df['VIX'] = vix['Close']
        df = df.dropna()

    # --- 2. INDICATORS ---
    window = 20
    df['SMA'] = df['Close'].rolling(window).mean()
    df['STD'] = df['Close'].rolling(window).std()
    df['Upper_BB2'] = df['SMA'] + (2 * df['STD'])
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

    plus_dm_series = pd.Series(df['PlusDM'].values, index=df.index)
    minus_dm_series = pd.Series(df['MinusDM'].values, index=df.index)
    
    df['PlusDI'] = 100 * (plus_dm_series.rolling(14).mean() / df['ATR'])
    df['MinusDI'] = 100 * (minus_dm_series.rolling(14).mean() / df['ATR'])
    
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
        
        current_equity = cash + (shares * price)
        equity_curve.append({'Date': date, 'Equity': current_equity})

        # Shark Bite (6% Rule)
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

        # Entry
        if shares == 0:
            price_condition = price < row['Lower_BB2']
            di_spread = row['MinusDI'] - row['PlusDI']
            di_condition = (row['MinusDI'] > row['PlusDI']) and (di_spread > di_threshold)
            vix_condition = (row['VIX'] > vix_min) and (row['VIX'] >= (row['VIX_High_20'] * vix_relative))

            if price_condition and di_condition and vix_condition:
                stop_level = row['Support_10']
                if stop_level >= price: stop_level = price * (1 - (stop_loss_pct/100))
                
                risk_per_share = price - stop_level
                risk_dollars = current_equity * (risk_per_trade/100)
                if risk_per_share <= 0: risk_per_share = 0.01 
                
                size = int(risk_dollars / risk_per_share)
                if size * price > cash: size = int(cash / price)

                if size > 0:
                    shares = size
                    cash -= shares * price
                    trade_log.append({'Date': date.date(), 'Type': 'BUY', 'Price': price, 'Shares': shares, 'PnL': 0})

        # Exit
        elif shares > 0:
            stop_line = row['Support_10']
            if row['Low'] < stop_line:
                cash += shares * stop_line
                trade_log.append({'Date': date.date(), 'Type': 'STOP LOSS', 'Price': stop_line, 'Shares': shares, 'PnL': -1})
                shares = 0
            else:
                should_sell = False
                exit_reason = ""
                if exit_strategy == "Middle Band (SMA)":
                    if price > df['SMA'][i]: should_sell = True; exit_reason = "TARGET (SMA)"
                elif exit_strategy == "Upper Band (2-Sigma)":
                    if price > row['Upper_BB2']: should_sell = True; exit_reason = "TARGET (UPPER BB)"
                elif exit_strategy == "Bullish DI Reversal (DI+ - DI- > 20)":
                    bull_spread = row['PlusDI'] - row['MinusDI']
                    if (row['PlusDI'] > row['MinusDI']) and (bull_spread > 20): should_sell = True; exit_reason = "TARGET (DI REVERSAL)"

                if should_sell:
                    cash += shares * price
                    trade_log.append({'Date': date.date(), 'Type': exit_reason, 'Price': price, 'Shares': shares, 'PnL': 1})
                    shares = 0

    # --- 4. CALCULATE QUANTCONNECT STYLE METRICS ---
    equity_df = pd.DataFrame(equity_curve).set_index('Date')
    
    # 1. Total Return
    final_val = equity_df['Equity'].iloc[-1]
    total_return = (final_val - initial_capital) / initial_capital
    
    # 2. CAGR (Compound Annual Growth Rate)
    days = (equity_df.index[-1] - equity_df.index[0]).days
    years = days / 365.25
    if years > 0:
        cagr = (final_val / initial_capital) ** (1 / years) - 1
    else:
        cagr = 0

    # 3. Max Drawdown
    equity_df['Peak'] = equity_df['Equity'].cummax()
    equity_df['Drawdown'] = (equity_df['Equity'] - equity_df['Peak']) / equity_df['Peak']
    max_drawdown = equity_df['Drawdown'].min()

    # 4. Sharpe Ratio (Risk Adjusted Return)
    equity_df['Daily_Return'] = equity_df['Equity'].pct_change()
    mean_daily_return = equity_df['Daily_Return'].mean()
    std_daily_return = equity_df['Daily_Return'].std()
    
    if std_daily_return != 0:
        sharpe_ratio = (mean_daily_return / std_daily_return) * np.sqrt(252)
    else:
        sharpe_ratio = 0

    # 5. Win Rate
    if trade_log:
        trades_df = pd.DataFrame(trade_log)
        # Filter for closed trades (sells)
        closed_trades = trades_df[trades_df['Type'].isin(['STOP LOSS', 'TARGET (SMA)', 'TARGET (UPPER BB)', 'TARGET (DI REVERSAL)', 'SHARK BITE'])]
        if not closed_trades.empty:
            wins = len(closed_trades[closed_trades['PnL'] > 0])
            total_closed = len(closed_trades)
            win_rate = wins / total_closed
        else:
            win_rate = 0
    else:
        win_rate = 0

    # --- 5. DISPLAY DASHBOARD ---
    
    st.subheader("Performance Overview")
    
    # Row 1: The Big Numbers
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Net Profit", f"${final_val - initial_capital:,.2f}", f"{total_return*100:.2f}%")
    col2.metric("CAGR", f"{cagr*100:.2f}%")
    col3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    col4.metric("Max Drawdown", f"{max_drawdown*100:.2f}%")

    # Row 2: Trade Stats
    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Win Rate", f"{win_rate*100:.1f}%")
    col6.metric("Total Trades", len([t for t in trade_log if t['Type'] == 'BUY']))
    col7.metric("Final Equity", f"${final_val:,.0f}")
    col8.metric("Years Tested", f"{years:.1f}")

    # Charts
    st.subheader("Equity Curve vs Drawdown")
    st.line_chart(equity_df[['Equity']])
    
    # Drawdown Chart (Area)
    st.area_chart(equity_df['Drawdown'])

    # Trade Log
    st.subheader("Trade Log")
    if trade_log:
        st.dataframe(trades_df)
    else:
        st.warning("No trades found with these settings.")

else:
    st.info("👈 Adjust settings in the sidebar and click 'Run Backtest'")
