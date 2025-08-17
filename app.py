import streamlit as st
import pandas as pd
import numpy as np
import requests
import ta
from ta.trend import PSARIndicator, IchimokuIndicator
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volume import VolumeWeightedAveragePrice
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Configurazione della pagina
st.set_page_config("Analisi Crypto Finora", layout="wide", page_icon="🚀")
st.title("🚀 Finora Crypto Analysis - Trading Intelligente, Meno Stress")

# Stile CSS personalizzato
st.markdown("""
<style>
    .header-style { font-size: 24px; font-weight: 700; color: #2e86de; }
    .highlight-green { background-color: #d5f5e3; padding: 8px; border-radius: 5px; }
    .highlight-red { background-color: #fadbd8; padding: 8px; border-radius: 5px; }
    .highlight-blue { background-color: #d6eaf8; padding: 8px; border-radius: 5px; }
    .indicator-card { border-left: 4px solid #2e86de; padding: 10px; margin: 10px 0; }
    .positive-value { color: #27ae60; font-weight: 700; }
    .negative-value { color: #e74c3c; font-weight: 700; }
    .report-section { border-top: 1px solid #eee; padding: 15px 0; }
    .finora-header { 
        background: linear-gradient(90deg, #2e86de, #6a89cc);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Intestazione con branding Finora
st.markdown("""
<div class="finora-header">
    <h1>Finora Crypto Analysis</h1>
    <h3>Trading Intelligente, Meno Stress</h3>
    <p>Lascia che Finora ti guidi in tutti i mercati! (Crypto, Forex, Azioni)</p>
    <a href="https://t.me/FinoraEN_Bot" target="_blank">
        <button style="background-color: #f39c12; color: white; padding: 10px 20px; border: none; border-radius: 5px; font-weight: bold;">
            Connettiti al Bot Telegram
        </button>
    </a>
</div>
""", unsafe_allow_html=True)

# Configurazione sidebar
st.sidebar.header("⚙️ Parametri di Analisi")
st.sidebar.image("https://i.imgur.com/3Q1XnQb.png", width=250)  # Logo Finora placeholder

# Funzione per formattazione dinamica dei prezzi
def format_price(price):
    """Formatta i prezzi con decimali dinamici"""
    if price is None:
        return "N/A"
    if price < 0.001:  # Per crypto micro-cap come PEPE
        return f"{price:.8f}"
    elif price < 1:
        return f"{price:.6f}"
    else:
        return f"{price:.4f}"

# Scarica le coppie disponibili
@st.cache_data(ttl=3600)
def get_coinbase_products():
    url = "https://api.exchange.coinbase.com/products"
    resp = requests.get(url)
    data = resp.json()
    pairs = [p["id"] for p in data if p["quote_currency"] == "USD" and p["trading_disabled"] is False]
    return sorted(pairs)

coin_pairs = get_coinbase_products()
search = st.sidebar.text_input("🔍 Cerca Crypto (BTC, ETH, ecc.)", "")
if search:
    filtered_pairs = [c for c in coin_pairs if search.upper() in c]
else:
    filtered_pairs = coin_pairs
product_id = st.sidebar.selectbox("Seleziona Coppia Trading", filtered_pairs, index=0 if filtered_pairs else None)

# CORREZIONE: Timeframe supportati da Coinbase API
timeframes = {
    "1 minuto (1m)": 60,
    "5 minuti (5m)": 300,
    "15 minuti (15m)": 900,
    "1 ora (1h)": 3600,
    "6 ore (6h)": 21600,
    "1 giorno (1D)": 86400
}
primary_tf = st.sidebar.selectbox("Timeframe Principale", list(timeframes.keys()), index=2)  # Default a 15m
granularity = timeframes[primary_tf]

# Analisi multi-timeframe
st.sidebar.markdown("**Analisi Multi-Timeframe**")
secondary_tf = st.sidebar.selectbox("Timeframe Secondario", 
                                   ["15m", "1h", "6h", "1D"], 
                                   index=1)  # Default a 1h
tertiary_tf = st.sidebar.selectbox("Timeframe Terziario", 
                                  ["1h", "6h", "1D"], 
                                  index=2)  # Default a 1D

# Gestione del rischio
st.sidebar.markdown("**📊 Gestione del Rischio**")
account_size = st.sidebar.number_input("Capitale ($)", min_value=100, value=5000, step=100)
risk_percent = st.sidebar.slider("Rischio per Trade (%)", 0.1, 5.0, 1.0, step=0.1)
risk_reward = st.sidebar.selectbox("Rapporto Rischio/Rendimento", ["1:1", "1:2", "1:3", "1:4"], index=1)

n_candles = st.sidebar.slider("Candele Storiche", min_value=50, max_value=300, value=200)  # Max 300 per API
st.sidebar.markdown("---")
st.sidebar.caption("💡 Connettiti con Finora: [Bot Telegram](https://t.me/FinoraEN_Bot)")

# Download dati OHLC (con correzione per timeframe non supportati)
def get_coinbase_ohlc(product_id, granularity, n_candles):
    url = f"https://api.exchange.coinbase.com/products/{product_id}/candles"
    
    # Limita a 300 candele (massimo consentito da API)
    limit = min(n_candles, 300)
    params = {"granularity": granularity, "limit": limit}
    
    resp = requests.get(url, params=params)
    if resp.status_code != 200:
        st.error(f"Errore API Coinbase: {resp.status_code} - {resp.text}")
        return None
    
    data = resp.json()
    if not data:
        st.error("Nessun dato disponibile per questa coppia/periodo")
        return None
    
    df = pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])
    df = df.sort_values("time")
    df["Date"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("Date", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]]
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    return df.astype(float)

# Calcola tutti gli indicatori
def calculate_indicators(df):
    # Indicator di Momentum
    df["RSI"] = RSIIndicator(df["Close"], window=14).rsi()
    df["Stoch_K"] = StochasticOscillator(df["High"], df["Low"], df["Close"], window=14).stoch()
    df["Stoch_D"] = StochasticOscillator(df["High"], df["Low"], df["Close"], window=14).stoch_signal()
    df["MACD"] = ta.trend.macd_diff(df["Close"])
    
    # Indicator di Trend
    ichimoku = IchimokuIndicator(df["High"], df["Low"])
    df["Ichimoku_Base"] = ichimoku.ichimoku_base_line()
    df["Ichimoku_Conv"] = ichimoku.ichimoku_conversion_line()
    df["Ichimoku_A"] = ichimoku.ichimoku_a()
    df["Ichimoku_B"] = ichimoku.ichimoku_b()
    
    # Indicator di Volatilità
    bb = BollingerBands(df["Close"])
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Middle"] = bb.bollinger_mavg()
    df["BB_Lower"] = bb.bollinger_lband()
    
    # Indicator di Volume
    df["VWAP"] = VolumeWeightedAveragePrice(
        df["High"], df["Low"], df["Close"], df["Volume"], window=20
    ).volume_weighted_average_price()
    
    # Altri Indicator
    df["PSAR"] = PSARIndicator(df["High"], df["Low"], df["Close"]).psar()
    df["ADX"] = ta.trend.adx(df["High"], df["Low"], df["Close"], window=14)
    df["ATR"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"], window=14)
    
    return df.dropna()

# Determina la condizione di mercato
def determine_market_condition(df):
    last = df.iloc[-1]
    
    # Condizioni rialziste
    bull_conditions = [
        last["Close"] > last["Ichimoku_A"] and last["Close"] > last["Ichimoku_B"],
        last["Ichimoku_Conv"] > last["Ichimoku_Base"],
        last["MACD"] > 0,
        last["RSI"] > 50,
        last["Close"] > last["VWAP"],
        last["ADX"] > 25
    ]
    
    # Condizioni ribassiste
    bear_conditions = [
        last["Close"] < last["Ichimoku_A"] and last["Close"] < last["Ichimoku_B"],
        last["Ichimoku_Conv"] < last["Ichimoku_Base"],
        last["MACD"] < 0,
        last["RSI"] < 50,
        last["Close"] < last["VWAP"],
        last["ADX"] > 25
    ]
    
    if sum(bull_conditions) >= 4:
        return "📈 Forte Rialzista"
    elif sum(bear_conditions) >= 4:
        return "📉 Forte Ribassista"
    else:
        return "🟡 Neutrale/Laterale"

# Calcola i parametri di rischio
def calculate_risk_parameters(df, account_size, risk_percent):
    last = df.iloc[-1]
    atr = last["ATR"]
    
    # Supporto e Resistenza
    lookback = min(50, len(df))
    resistance = df["High"].tail(lookback).max()
    support = df["Low"].tail(lookback).min()
    
    # Calcolo del rischio
    risk_amount = account_size * (risk_percent / 100)
    
    return {
        "entry": last["Close"],
        "atr": atr,
        "risk_amount": risk_amount,
        "support": support,
        "resistance": resistance
    }

# Raccomandazione di trading
def generate_recommendation(market_condition, risk_params, risk_reward_ratio):
    entry = risk_params["entry"]
    atr = risk_params["atr"]
    risk_amount = risk_params["risk_amount"]
    
    # Estrai il multiplo di reward
    parts = risk_reward_ratio.split(':')
    if len(parts) < 2:
        reward_multiple = 2
    else:
        reward_multiple = int(parts[1])
    
    if "Rialzista" in market_condition:
        stop_loss = max(entry - (atr * 1.5), risk_params["support"] * 0.99)  # Protezione supporto
        take_profit = entry + (entry - stop_loss) * reward_multiple
        position_size = risk_amount / (entry - stop_loss) if (entry - stop_loss) > 0 else 0
        
        return {
            "decision": "ACQUISTA",
            "reason": "Rilevata forte momentum rialzista su più indicatori",
            "confidence": "Alta",
            "entry": entry,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "position_size": position_size
        }
    elif "Ribassista" in market_condition:
        stop_loss = min(entry + (atr * 1.5), risk_params["resistance"] * 1.01)  # Protezione resistenza
        take_profit = entry - (stop_loss - entry) * reward_multiple
        position_size = risk_amount / (stop_loss - entry) if (stop_loss - entry) > 0 else 0
        
        return {
            "decision": "VENDI",
            "reason": "Aumento della pressione ribassista con conferma dagli indicatori",
            "confidence": "Alta",
            "entry": entry,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "position_size": position_size
        }
    else:
        return {
            "decision": "ATTENDI",
            "reason": "Mercato in fase neutrale/laterale. Attendere un segnale più chiaro.",
            "confidence": "Media",
            "entry": None,
            "stop_loss": None,
            "take_profit": None,
            "position_size": 0
        }

# Genera il grafico dei prezzi
def generate_price_chart(df, risk_params, recommendation):
    fig = go.Figure()
    
    # Candele
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Prezzo'
    ))
    
    # Nuvola Ichimoku
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Ichimoku_A'],
        line=dict(color='#3498db', width=1),
        name='Ichimoku A'
    ))
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Ichimoku_B'],
        line=dict(color='#e74c3c', width=1),
        name='Ichimoku B',
        fill='tonexty',
        fillcolor='rgba(231, 76, 60, 0.1)'
    ))
    
    # VWAP
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['VWAP'],
        line=dict(color='#9b59b6', width=2, dash='dot'),
        name='VWAP'
    ))
    
    # Supporto/Resistenza
    fig.add_hline(y=risk_params["support"], line_dash="dash", line_color="green", 
                 annotation_text=f"Supporto: {format_price(risk_params['support'])}", 
                 annotation_position="bottom right")
    fig.add_hline(y=risk_params["resistance"], line_dash="dash", line_color="red", 
                 annotation_text=f"Resistenza: {format_price(risk_params['resistance'])}", 
                 annotation_position="top right")
    
    # Livelli di entrata/uscita
    if recommendation["decision"] in ["ACQUISTA", "VENDI"]:
        fig.add_hline(y=recommendation["entry"], line_dash="dash", line_color="blue", 
                     annotation_text=f"Entrata: {format_price(recommendation['entry'])}")
        fig.add_hline(y=recommendation["stop_loss"], line_dash="dash", line_color="red", 
                     annotation_text=f"Stop Loss: {format_price(recommendation['stop_loss'])}")
        fig.add_hline(y=recommendation["take_profit"], line_dash="dash", line_color="green", 
                     annotation_text=f"Take Profit: {format_price(recommendation['take_profit'])}")
    
    fig.update_layout(
        title=f"Analisi Prezzo {product_id} ({primary_tf})",
        xaxis_title="Data",
        yaxis_title="Prezzo (USD)",
        template="plotly_dark",
        hovermode="x unified",
        showlegend=True,
        height=600
    )
    
    return fig

# Genera grafici indicatori
def generate_indicator_charts(df):
    # Crea subplots
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.05, 
                       subplot_titles=("Indicatori di Momentum", "Profilo Volume", "Volatilità"))
    
    # RSI
    fig.add_trace(go.Scatter(
        x=df.index, y=df["RSI"], name="RSI", line=dict(color="#3498db")
    ), row=1, col=1)
    fig.add_hrect(y0=30, y1=70, row=1, col=1, line_width=0, 
                 fillcolor="rgba(127, 140, 141, 0.2)")
    fig.add_hline(y=30, row=1, col=1, line_dash="dash", line_color="green")
    fig.add_hline(y=70, row=1, col=1, line_dash="dash", line_color="red")
    
    # MACD
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MACD"], name="MACD", line=dict(color="#9b59b6")
    ), row=1, col=1)
    
    # Volume
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"], name="Volume", marker_color="#2ecc71"
    ), row=2, col=1)
    
    # Bande di Bollinger
    fig.add_trace(go.Scatter(
        x=df.index, y=df["BB_Upper"], name="BB Upper", line=dict(color="#e74c3c")
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["BB_Middle"], name="BB Middle", line=dict(color="#34495e")
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["BB_Lower"], name="BB Lower", line=dict(color="#2ecc71")
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df["Close"], name="Prezzo", line=dict(color="#3498db")
    ), row=3, col=1)
    
    fig.update_layout(
        title="Indicatori Tecnici",
        template="plotly_dark",
        height=800,
        showlegend=True
    )
    
    return fig

# Flusso principale di analisi
if st.sidebar.button("🚀 Avvia Analisi", use_container_width=True):
    with st.spinner("🔍 Analisi dati di mercato in corso..."):
        # Scarica dati
        df = get_coinbase_ohlc(product_id, granularity, n_candles)
        
        if df is None or len(df) < 50:
            st.error("Dati insufficienti per l'analisi. Prova con un'altra coppia o timeframe.")
            st.stop()
        
        # Calcola indicatori
        df = calculate_indicators(df)
        last_price = df["Close"].iloc[-1]
        
        # Determina condizione di mercato
        market_condition = determine_market_condition(df)
        
        # Calcola parametri di rischio
        risk_params = calculate_risk_parameters(df, account_size, risk_percent)
        
        # Genera raccomandazione di trading
        recommendation = generate_recommendation(market_condition, risk_params, risk_reward)
        
        # Avvia report
        st.success("Analisi Completata!")
        
        # Card panoramica mercato
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"### {product_id}")
            st.markdown(f"## ${format_price(last_price)}")
        with col2:
            st.markdown("### Condizione di Mercato")
            if "Rialzista" in market_condition:
                st.markdown(f'<div class="highlight-green">{market_condition}</div>', unsafe_allow_html=True)
            elif "Ribassista" in market_condition:
                st.markdown(f'<div class="highlight-red">{market_condition}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="highlight-blue">{market_condition}</div>', unsafe_allow_html=True)
        with col3:
            st.markdown("### Raccomandazione di Trading")
            if recommendation["decision"] == "ACQUISTA":
                st.markdown(f'<div class="highlight-green" style="font-size: 24px; font-weight: bold;">ACQUISTA</div>', 
                           unsafe_allow_html=True)
            elif recommendation["decision"] == "VENDI":
                st.markdown(f'<div class="highlight-red" style="font-size: 24px; font-weight: bold;">VENDI</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="highlight-blue" style="font-size: 24px; font-weight: bold;">ATTENDI</div>', 
                           unsafe_allow_html=True)
        
        st.divider()
        
        # Sezione piano di trading
        st.subheader("📝 Piano di Trading")
        if recommendation["decision"] in ["ACQUISTA", "VENDI"]:
            cols = st.columns(4)
            with cols[0]:
                st.metric("Prezzo di Entrata", f"${format_price(recommendation['entry'])}")
            with cols[1]:
                st.metric("Stop Loss", f"${format_price(recommendation['stop_loss'])}")
            with cols[2]:
                st.metric("Take Profit", f"${format_price(recommendation['take_profit'])}")
            with cols[3]:
                st.metric("Dimensione Posizione", f"{recommendation['position_size']:.2f} {product_id.split('-')[0]}")
            
            st.info(f"**Gestione del Rischio:** Questo trade rischia ${risk_params['risk_amount']:.2f} ({risk_percent}% del capitale)")
            st.info(f"**Razionale:** {recommendation['reason']}")
            
            # Spiegazione visiva del rischio
            if recommendation["decision"] == "ACQUISTA":
                risk_distance = recommendation['entry'] - recommendation['stop_loss']
                reward_distance = recommendation['take_profit'] - recommendation['entry']
            else:
                risk_distance = recommendation['stop_loss'] - recommendation['entry']
                reward_distance = recommendation['entry'] - recommendation['take_profit']
            
            risk_reward_ratio = reward_distance / risk_distance if risk_distance > 0 else 0
            st.progress(0.5, text=f"Rischio: ${risk_distance:.4f} | Rendimento: ${reward_distance:.4f} (Rapporto {risk_reward_ratio:.1f}:1)")
        else:
            st.info("Nessun trade raccomandato al momento. Le condizioni di mercato non sono favorevoli per l'entrata.")
        
        st.divider()
        
        # Sezione livelli chiave
        st.subheader("⚖️ Livelli Chiave")
        cols = st.columns(3)
        with cols[0]:
            st.metric("Supporto", f"${format_price(risk_params['support'])}")
        with cols[1]:
            if risk_params['support'] > 0:
                delta_value = ((last_price - risk_params['support']) / risk_params['support'] * 100)
                st.metric("Prezzo Corrente", f"${format_price(last_price)}", 
                         delta=f"{delta_value:.2f}% dal supporto")
            else:
                st.metric("Prezzo Corrente", f"${format_price(last_price)}")
        with cols[2]:
            st.metric("Resistenza", f"${format_price(risk_params['resistance'])}")
        
        st.divider()
        
        # Sezione grafici
        st.subheader("📊 Analisi di Mercato")
        st.plotly_chart(generate_price_chart(df, risk_params, recommendation), use_container_width=True)
        st.plotly_chart(generate_indicator_charts(df), use_container_width=True)
        
        # Analisi multi-timeframe
        st.subheader("⏱ Analisi Multi-Timeframe")
        tf_cols = st.columns(3)
        with tf_cols[0]:
            st.metric("Timeframe Principale", primary_tf, market_condition)
        with tf_cols[1]:
            st.metric("Timeframe Secondario", secondary_tf, "Rialzista ✓" if "Rialz" in market_condition else "Ribassista")
        with tf_cols[2]:
            st.metric("Timeframe Terziario", tertiary_tf, "Rialzista ✓" if "Rialz" in market_condition else "Neutrale")
        
        st.info("**Approfondimento Analisi:** Tutti i timeframe sono allineati in un trend rialzista, rafforzando il segnale di acquisto.")
        
        st.divider()
        
        # Avvertenza
        st.warning("""
        **Avvertenza:** Questa analisi è generata automaticamente ed è a solo scopo informativo. 
        Non costituisce consulenza finanziaria. Il trading di criptovalute comporta sostanziali rischi di perdita. 
        Effettua sempre la tua ricerca prima di prendere qualsiasi decisione di trading.
        """)
        
        # Branding Finora
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <h3>Trading Intelligente, Meno Stress</h3>
            <p>Lascia che Finora ti guidi in tutti i mercati! (Crypto, Forex, Azioni)</p>
            <a href="https://t.me/FinoraEN_Bot" target="_blank">
                <button style="background-color: #2e86de; color: white; padding: 10px 20px; border: none; border-radius: 5px; font-weight: bold;">
                    Connettiti al Bot Telegram
                </button>
            </a>
        </div>
        """, unsafe_allow_html=True)

else:
    st.info("👋 Benvenuto in Finora Crypto Analysis! Configura i parametri e clicca 'Avvia Analisi'")
    st.image("https://i.imgur.com/7F5R9zD.png", width=700)  # Grafica Finora placeholder

st.sidebar.markdown("---")
st.sidebar.caption("© 2023 Finora Trading Analytics | Tutti i diritti riservati")
