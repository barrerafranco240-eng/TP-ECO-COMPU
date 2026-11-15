import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import feedparser
from transformers import pipeline

# Configuración general de la página
st.set_page_config(page_title="Monitor de Inversiones", page_icon="📈", layout="wide")

# --- FUNCIONES CON CACHÉ (Vital para que la web sea rápida) ---
@st.cache_resource
def cargar_modelo_ia():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

@st.cache_data(ttl=300)  # Guarda los datos por 5 minutos para no saturar Yahoo Finance
def descargar_datos_activo(ticker):
    df = yf.download(ticker, period="3mo", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

@st.cache_data(ttl=300)
def obtener_noticias_rss(ticker):
    url_rss = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US"
    feed = feedparser.parse(url_rss)
    return [entrada.title for entrada in feed.entries[:5]]

# --- PANEL LATERAL: GESTIÓN INTERACTIVA DE CARTERA ---
st.sidebar.header("💼 Mi Cartera de Activos")
st.sidebar.write("Editá la columna **Cantidad** para definir tu tenencia actual (o agregá filas nuevas):")

# Cartera inicial por defecto
cartera_inicial = pd.DataFrame({
    "Activo": ["SPY", "AAPL", "MSFT", "NVDA", "AMZN", "META", "YPF", "GGAL", "BTC-USD"],
    "Cantidad": [5.0, 10.0, 0.0, 4.0, 0.0, 0.0, 15.0, 0.0, 0.0]
})

# Tabla interactiva editable
cartera_editada = st.sidebar.data_editor(
    cartera_inicial,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True
)

# Limpiamos filas vacías si el usuario agregó una fila sin completar
cartera_editada = cartera_editada.dropna(subset=["Activo"])
cartera_editada["Activo"] = cartera_editada["Activo"].str.upper().str.strip()
cartera_editada["Cantidad"] = pd.to_numeric(cartera_editada["Cantidad"], errors="coerce").fillna(0.0)

st.sidebar.divider()

# Selector del activo a analizar en profundidad
lista_activos = cartera_editada["Activo"].tolist()
activo = st.sidebar.selectbox("🔍 Elegí el activo para auditar:", lista_activos)

# Obtenemos automáticamente la cantidad que el usuario puso en la tabla para ese activo
cantidad_tenencia = float(cartera_editada.loc[cartera_editada["Activo"] == activo, "Cantidad"].iloc[0])

# --- CUERPO PRINCIPAL DE LA APLICACIÓN ---
st.title("📈 Monitor de Inversiones & Radar Bursátil")
st.markdown("Análisis cuantitativo en tiempo real, sentimiento de noticias con IA (BERT) y gestión de posición.")

# --- SECCIÓN 1: RADAR DE OPORTUNIDADES (SCREENER) ---
with st.expander("📡 Ver Radar de Mercado (Top Gainer de tu lista)", expanded=True):
    if st.button("Escanear rendimientos diarios de mi lista"):
        mejor_activo = None
        mejor_rendimiento = 0.0
        
        with st.spinner("Escaneando cotizaciones..."):
            for t in lista_activos:
                try:
                    d = yf.download(t, period="5d", progress=False)
                    if isinstance(d.columns, pd.MultiIndex):
                        d.columns = d.columns.get_level_values(0)
                    cierres = d["Close"].dropna()
                    if len(cierres) >= 2:
                        var = ((float(cierres.iloc[-1]) - float(cierres.iloc[-2])) / float(cierres.iloc[-2])) * 100
                        if var > mejor_rendimiento:
                            mejor_rendimiento = var
                            mejor_activo = t
                except Exception:
                    continue
                    
        if mejor_activo:
            st.success(f"🟢 **OPORTUNIDAD DETECTADA:** **{mejor_activo}** lidera tu lista hoy con una suba de **+{mejor_rendimiento:.2f}%**.")
        else:
            st.warning("🟡 Ninguno de los activos de tu lista presenta subas hoy.")

st.divider()

# --- SECCIÓN 2: ANÁLISIS DEL ACTIVO SELECCIONADO ---
datos_mercado = descargar_datos_activo(activo)

if datos_mercado.empty:
    st.error(f"No se encontraron datos para el ticker '{activo}'. Verificá que esté bien escrito según Yahoo Finance.")
    st.stop()

# Procesamiento con Pandas
datos_mercado.rename(columns={"Close": "Cierre"}, inplace=True)
datos_mercado["Retorno_Diario"] = datos_mercado["Cierre"].pct_change() * 100
datos_mercado["Media_Movil_7d"] = datos_mercado["Cierre"].rolling(window=7).mean()
df_limpio = datos_mercado[["Cierre", "Retorno_Diario", "Media_Movil_7d"]].dropna()

precio_hoy = float(df_limpio["Cierre"].iloc[-1])
media_movil_hoy = float(df_limpio["Media_Movil_7d"].iloc[-1])
valor_posicion = cantidad_tenencia * precio_hoy
dif_mm = ((precio_hoy - media_movil_hoy) / media_movil_hoy) * 100

# Tarjetas ejecutivas (KPIs)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Cotización Actual", f"${precio_hoy:,.2f}", f"{df_limpio['Retorno_Diario'].iloc[-1]:.2f}% hoy")
col2.metric("Media Móvil (7d)", f"${media_movil_hoy:,.2f}", f"{dif_mm:+.2f}% vs precio")
col3.metric("Tu Tenencia", f"{cantidad_tenencia:g} unid.")
col4.metric("Valuación Posición", f"${valor_posicion:,.2f} USD")

# Columnas para Gráfico + IA
col_izq, col_der = st.columns([1.4, 1])

with col_izq:
    st.subheader(f"📊 Tendencia Técnica: {activo}")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.set_theme(style="darkgrid")
    ax.plot(df_limpio.index, df_limpio["Cierre"], label="Precio de Cierre (USD)", color="#1f77b4", linewidth=2)
    ax.plot(df_limpio.index, df_limpio["Media_Movil_7d"], label="Media Móvil (7 días)", color="#ff7f0e", linestyle="--", linewidth=2)
    ax.set_title(f"Evolución de {activo} (Últimos 3 meses)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Precio (USD)")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

with col_der:
    st.subheader("🧠 Sentimiento de Noticias (NLP)")
    titulos = obtener_noticias_rss(activo)
    
    if titulos:
        for t in titulos:
            st.markdown(f"- {t}")
        
        reporte_diario = ". ".join(titulos)
        with st.spinner("Evaluando contexto con red neuronal BERT..."):
            analizador = cargar_modelo_ia()
            resultado_ia = analizador(reporte_diario[:512])  # Límite seguro de tokens para BERT
            etiqueta = resultado_ia[0]["label"]
            confianza = resultado_ia[0]["score"]
            estrellas_ia = int(etiqueta.split()[0])
            
        st.info(f"**Diagnóstico IA:** {etiqueta} | **Certeza:** {confianza:.1%}")
    else:
        st.warning("Sin titulares RSS recientes para este ticker. Se asume sentimiento neutral (3 estrellas).")
        estrellas_ia = 3

# --- SECCIÓN 3: VEREDICTO AUTOMÁTICO SEGÚN TENENCIA ---
st.subheader("🤖 Veredicto Algorítmico de Inversión")

mercado_alcista = (precio_hoy > media_movil_hoy) and (estrellas_ia >= 4)
mercado_bajista = (precio_hoy < media_movil_hoy) and (estrellas_ia <= 2)

if cantidad_tenencia == 0:
    if mercado_alcista:
        st.success("🟢 **COMPRAR (Abrir posición):** No tenés tenencia y hay convergencia alcista (Precio > Media Móvil + Noticias optimistas).")
    elif mercado_bajista:
        st.error("🔴 **EVITAR ENTRADA (Mantener liquidez):** Tendencia bajista y noticias negativas. Al tener 0 unidades, se bloquean compras.")
    else:
        st.warning("🟡 **ESPERAR (En observación):** Señales mixtas entre el gráfico técnico y las noticias. Sin confirmación para entrar.")
else:
    if mercado_alcista:
        st.success(f"🟢 **MANTENER O ACUMULAR:** Tu posición actual (${valor_posicion:,.2f} USD) acompaña una tendencia alcista validada por la IA.")
    elif mercado_bajista:
        st.error(f"🔴 **VENDER (Liquidar las {cantidad_tenencia:g} unidades):** Ruptura bajo la media móvil confirmada por noticias pesimistas. Ejecutar toma de ganancias o Stop-Loss.")
    else:
        st.warning(f"🟡 **MANTENER (Hold):** Mercado sin tendencia definida. Conviene conservar tus {cantidad_tenencia:g} unidades y evitar costos de transacción.")
