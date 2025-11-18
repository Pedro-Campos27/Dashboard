# app.py — Dashboard de Temperatura (CSV, multi-dia, multi-dispositivo)
# Run: streamlit run app.py

import io, os, re
from datetime import datetime, timedelta
from typing import Optional, Dict, List

import pytz
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================
TZ = "America/Sao_Paulo"
LOCAL_TZ = pytz.timezone(TZ)

DEFAULT_PERIOD = "1T"      # 1, 5, 15, 30, 60 minutos (usar 'T' do pandas)
DEFAULT_STAT = "mean"      # mean|min|max|count

# (Opcional) Mapeie padrões de nome de arquivo => nome do sensor
# Ex.: r"^ConservadoraA__.*": "Conservadora A"
REGEX_SENSOR_MAP: Dict[str, str] = {
    # r"^ConservadoraA__.*": "Conservadora A",
    # r"^FreezerX__.*": "Freezer X",
}

# =============================================================================
# UI
# =============================================================================
st.set_page_config(page_title="Dashboard (CSV) — Temperatura", layout="wide")
st.title("Dashboard de Temperatura (via CSV)")
st.caption("Consome 1..N CSVs por dispositivo (um CSV por dia), mescla por dispositivo e plota linhas contínuas multi-dia.")

hoje = datetime.now(LOCAL_TZ).date()
with st.sidebar:
    st.header("Arquivos CSV")
    st.caption("Esperado: `timestamp` (epoch ms) e/ou `date` (America/Sao_Paulo) + coluna de temperatura (ex.: `temperatura_pr1`).")
    files = st.file_uploader("Selecione 1..N CSVs", type=["csv"], accept_multiple_files=True)

    st.subheader("Período")
    auto_window = st.toggle("Detectar período pelos CSVs", value=True)
    start_date = st.date_input("Início", hoje - timedelta(days=1), disabled=auto_window)
    end_date = st.date_input("Fim", hoje, disabled=auto_window)

    st.subheader("Agregação")
    period = st.selectbox("Janela", ["1T", "5T", "15T", "30T", "60T"],
                          index=["1T","5T","15T","30T","60T"].index(DEFAULT_PERIOD))
    statistic = st.selectbox("Estatística", ["mean", "min", "max", "count"],
                             index=["mean","min","max","count"].index(DEFAULT_STAT))

    st.subheader("Qualidade")
    zscore_filter = st.toggle("Filtrar outliers (|z|>3)", value=True)
    interpolate_gaps = st.toggle("Interpolar buracos pequenos (≤5 min)", value=False)

    st.subheader("Faixa ANVISA")
    anvisa_min = st.number_input("Mínimo (°C)", value=2.0, step=0.1)
    anvisa_max = st.number_input("Máximo (°C)", value=8.0, step=0.1)

    st.subheader("Plotagem")
    plot_all_together = st.toggle("Plotar todos os dispositivos no mesmo gráfico", value=True)

    run = st.button("Processar & Plotar", type="primary")

# =============================================================================
# FUNÇÕES
# =============================================================================
def infer_sensor_name(file_name: str) -> str:
    """
    Deduz o nome do sensor a partir do nome do arquivo:
    1) Se bater com algum REGEX_SENSOR_MAP -> usa o mapeado.
    2) Senão, remove sufixos de data (ex.: _2025-11-16, -20251117, _dia16, 16-11-2025).
    """
    base = os.path.splitext(file_name)[0]

    # 1) regex explícito (se configurado)
    for pat, nome in REGEX_SENSOR_MAP.items():
        if re.match(pat, base, flags=re.IGNORECASE):
            return nome

    # 2) heurística simples pra remover datas do final do nome
    patterns = [
        r"[\-_]?\d{4}[\-_]?\d{2}[\-_]?\d{2}$",   # 2025-11-16 | 20251116 | 2025_11_16
        r"[\-_]?\d{8}$",                         # 20251116
        r"[\-_]?dia\d{1,2}$",                    # dia16
        r"[\-_]?\d{2}[\-_]\d{2}[\-_]\d{4}$",     # 16-11-2025
    ]
    for pat in patterns:
        base = re.sub(pat, "", base, flags=re.IGNORECASE)
    return base.strip(" _-")

def _pick_value_column(df: pd.DataFrame) -> Optional[str]:
    """
    Tenta achar a coluna de temperatura:
    - preferência por colunas com 'temp' ou 'temperatura' no nome;
    - senão, primeira numérica que não seja 'timestamp'.
    """
    candidates = [c for c in df.columns if re.search(r"temp|temperatura", c, re.I)]
    candidates = [c for c in candidates if c.lower() not in ("timestamp", "date")]
    if candidates:
        return candidates[0]
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c.lower() not in ("timestamp",)]
    return num_cols[0] if num_cols else None

def _parse_timestamp(df: pd.DataFrame) -> pd.Series:
    """
    Retorna série tz-aware em America/Sao_Paulo:
    - se existir 'date' (string): parse e ajusta tz (localiza ou converte);
    - senão, usa 'timestamp' (epoch ms UTC) e converte pro TZ alvo.
    """
    if "date" in df.columns:
        s = pd.to_datetime(df["date"], errors="coerce")
        s = s.dt.tz_localize(TZ) if s.dt.tz is None else s.dt.tz_convert(TZ)
        return s
    if "timestamp" in df.columns:
        s = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce").dt.tz_convert(TZ)
        return s
    raise ValueError("CSV sem colunas 'date' nem 'timestamp'.")

def read_csv_raw(file, preferred_value_col: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(file)

    # timestamp -> tz-aware America/Sao_Paulo
    ts = _parse_timestamp(df)

    # detecta coluna de temperatura
    value_col = preferred_value_col or _pick_value_column(df)
    if not value_col:
        raise ValueError(
            "Não consegui detectar a coluna de temperatura. "
            "Nomeie algo como 'temperatura_*' ou 'temp_*', ou escolha manualmente no fallback."
        )

    # limpa e força numérico (virgula decimal, espaços, etc.)
    vals = (
        df[value_col]
        .astype(str)
        .str.strip()
        .str.replace(",", ".", regex=False)
    )
    vals = pd.to_numeric(vals, errors="coerce")

    out = (
        pd.DataFrame({"timestamp": ts, "value": vals})
        .dropna(subset=["timestamp", "value"])
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return out


def resample_stat_per_sensor(df_all: pd.DataFrame, period: str, stat: str) -> pd.DataFrame:
    out: List[pd.DataFrame] = []

    for sensor in df_all["sensor"].unique():
        cur = df_all[df_all["sensor"] == sensor][["timestamp", "value"]].copy()

        # garante datetime index e numerico
        if not pd.api.types.is_datetime64_any_dtype(cur["timestamp"]):
            cur["timestamp"] = pd.to_datetime(cur["timestamp"], errors="coerce")
        cur["value"] = pd.to_numeric(cur["value"], errors="coerce")

        cur = cur.dropna(subset=["timestamp", "value"]).sort_values("timestamp")
        if cur.empty:
            continue

        cur = cur.set_index("timestamp")

        # aplica a estatística na SÉRIE (evita dtype object do DF com colunas não numéricas)
        resampler = cur["value"].resample(period)
        if stat == "mean":
            r = resampler.mean()
        elif stat == "min":
            r = resampler.min()
        elif stat == "max":
            r = resampler.max()
        elif stat == "count":
            r = resampler.count()
        else:
            r = resampler.mean()

        r = r.dropna()
        if r.empty:
            continue

        r = r.to_frame(name="value").reset_index()
        r["sensor"] = sensor
        out.append(r)

    if not out:
        return pd.DataFrame(columns=["timestamp", "value", "sensor"])

    return pd.concat(out, ignore_index=True)

def apply_zscore_filter(df: pd.DataFrame, col="value", threshold=3.0):
    """
    Remove outliers por sensor usando z-score simétrico.
    """
    if df.empty:
        return df, 0
    kept, removed = [], 0
    for sensor in df["sensor"].unique():
        cur = df[df["sensor"] == sensor].copy()
        x = cur[col].astype(float)
        std = x.std(ddof=1)
        z = (x - x.mean()) / (std if std > 0 else 1.0)
        m = z.abs() <= threshold
        removed += int((~m).sum())
        kept.append(cur[m])
    return pd.concat(kept, ignore_index=True), removed

def interpolate_small_gaps(df: pd.DataFrame, freq_str: str, limit_minutes=5):
    out = []
    step_minutes = max(1, int(pd.to_timedelta(freq_str).total_seconds() // 60))
    limit = max(1, limit_minutes // step_minutes)

    for sensor in df["sensor"].unique():
        cur = df[df["sensor"] == sensor][["timestamp", "value"]].copy()
        cur = cur.set_index("timestamp").asfreq(freq_str)
        cur["value"] = pd.to_numeric(cur["value"], errors="coerce")
        cur["value"] = cur["value"].interpolate(limit=limit, limit_area="inside")
        cur = cur.reset_index()
        cur["sensor"] = sensor
        out.append(cur)

    return pd.concat(out, ignore_index=True)


def kpi_completude(df: pd.DataFrame, start_dt: pd.Timestamp, end_dt: pd.Timestamp, period: str):
    if df.empty:
        return 0.0, 0, 0, pd.to_timedelta(0), pd.to_timedelta(0)
    freq = pd.to_timedelta(period)
    expected = int(((end_dt - start_dt).total_seconds() // freq.total_seconds()) + 1)
    got = df["timestamp"].nunique()
    completeness = (got / expected) * 100.0 if expected > 0 else 0.0
    s = df["timestamp"].sort_values().reset_index(drop=True)
    deltas = s.diff().dropna()
    gap_threshold = pd.to_timedelta(period) * 1.5
    gaps = deltas[deltas > gap_threshold]
    total_gap = gaps.sum() if not gaps.empty else pd.to_timedelta(0)
    max_gap = gaps.max() if not gaps.empty else pd.to_timedelta(0)
    return round(completeness, 2), expected, got, total_gap, max_gap

def kpi_anvisa(df: pd.DataFrame, lo: float, hi: float):
    if df.empty:
        return 0, 0, pd.to_timedelta(0), pd.to_timedelta(0), 0, pd.to_timedelta(0)
    s = df["timestamp"].sort_values().reset_index(drop=True)
    step = s.diff().mode().iloc[0] if len(s) >= 2 else pd.to_timedelta("1T")
    mask_in = (df["value"] >= lo) & (df["value"] <= hi)
    dur_in = mask_in.sum() * step
    dur_out = (~mask_in).sum() * step
    out_blocks, max_out, cur = 0, pd.to_timedelta(0), pd.to_timedelta(0)
    for ok in mask_in:
        if not ok:
            cur += step
        else:
            if cur > pd.to_timedelta(0):
                out_blocks += 1
                max_out = max(max_out, cur)
            cur = pd.to_timedelta(0)
    if cur > pd.to_timedelta(0):
        out_blocks += 1
        max_out = max(max_out, cur)
    return mask_in.sum(), (~mask_in).sum(), dur_in, dur_out, out_blocks, max_out

def fmt_td(td: pd.Timedelta) -> str:
    secs = int(td.total_seconds())
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

# =============================================================================
# PIPELINE
# =============================================================================
if run:
    if not files:
        st.error("Envie pelo menos um CSV.")
        st.stop()

    # 1) LER TODOS OS CSVs CRUS E ETIQUETAR SENSOR POR NOME-BASE
    raw_frames = []
    with st.status("Lendo CSV(s)…", expanded=False):
        for f in files:
            try:
                df_raw = read_csv_raw(f)
            except ValueError as e:
                # Fallback interativo: seleciona manualmente a coluna de valor
                st.warning(f"{f.name}: {e}")
                df_full = pd.read_csv(f)
                options = [c for c in df_full.columns if c.lower() not in ("timestamp", "date")]
                with st.expander(f"Escolha a coluna de temperatura para {f.name}", expanded=True):
                    picked = st.selectbox(f"Coluna de valor – {f.name}", options=options, key=f"pick_{f.name}")
                df_raw = read_csv_raw(io.StringIO(df_full.to_csv(index=False)), preferred_value_col=picked)

            df_raw["sensor"] = infer_sensor_name(f.name)
            raw_frames.append(df_raw)

    if not raw_frames:
        st.warning("Sem dados após leitura.")
        st.stop()

    df_raw_all = (
        pd.concat(raw_frames, ignore_index=True)
          .dropna()
          .sort_values(["sensor", "timestamp"])
          .reset_index(drop=True)
    )

    # 2) JANELA AUTOMÁTICA OU MANUAL
    if auto_window:
        start_dt = df_raw_all["timestamp"].min()
        end_dt = df_raw_all["timestamp"].max()
    else:
        start_dt = LOCAL_TZ.localize(datetime.combine(start_date, datetime.min.time()))
        end_dt = LOCAL_TZ.localize(datetime.combine(end_date, datetime.max.time()))
        if start_dt > end_dt:
            st.error("Intervalo inválido.")
            st.stop()

    # 3) FILTRAR PELA JANELA E SÓ ENTÃO RESAMPLE POR SENSOR
    df_filtrado = df_raw_all[(df_raw_all["timestamp"] >= start_dt) & (df_raw_all["timestamp"] <= end_dt)]
    df_all = resample_stat_per_sensor(df_filtrado, period, statistic)
    if df_all.empty:
        st.warning("Sem dados na janela selecionada.")
        st.stop()

    # 4) QUALIDADE
    removed = 0
    if zscore_filter:
        df_all, removed = apply_zscore_filter(df_all, "value", 3.0)

    if interpolate_gaps:
        df_all = interpolate_small_gaps(df_all, freq_str=period, limit_minutes=5)
        df_all = df_all.dropna(subset=["value"])

    # 5) SELEÇÃO DE DISPOSITIVOS (pra overlay ficar limpinho)
    sensores_disponiveis = sorted(df_all["sensor"].unique().tolist())
   # st.multiselect.__doc__  # só pra agradar linters
    selected_sensors = st.multiselect(
        "Selecione os dispositivos a exibir (aplica em todas as abas abaixo):",
        options=sensores_disponiveis,
        default=sensores_disponiveis,
        key="sel_sensors_top",
    )
    df_all = df_all[df_all["sensor"].isin(selected_sensors)]
    if df_all.empty:
        st.warning("Nada a exibir após filtragem por dispositivo.")
        st.stop()

    # =============================================================================
    # TABS
    # =============================================================================
    tab_kpi, tab_series, tab_heat, tab_hist, tab_raw = st.tabs(
        ["KPIs", "Séries & Faixa", "Mapa de Calor", "Distribuição", "Dados Brutos"]
    )

    # ------------------------- KPIs -------------------------
    with tab_kpi:
     st.markdown("### Visão geral (por dispositivo)")
    # Header compacto do contexto da consulta
    c0, c1, c2, c3, c4 = st.columns([2,1,1,1,1])
    c0.write(f"**Período:** {start_dt:%Y-%m-%d %H:%M} → {end_dt:%Y-%m-%d %H:%M}")
    c1.metric("Amostragem", f"{statistic}/{period}")
    c2.metric("Timezone", TZ)
    c3.metric("Faixa alvo", f"{anvisa_min:.1f}–{anvisa_max:.1f} °C")
    c4.metric("Outliers removidos", f"{removed}")

    st.divider()

    def classifica_status(pct_fora: float, maior_excursao: pd.Timedelta) -> tuple[str, str]:
        """
        Regras simples e objetivas:
        - OK: fora <= 1% e maior excursão <= 5 min
        - ATENÇÃO: fora <= 5% ou maior excursão <= 15 min
        - CRÍTICO: acima disso
        """
        if pct_fora <= 1.0 and maior_excursao <= pd.to_timedelta("5min"):
            return "OK", "🟢"
        if pct_fora <= 5.0 or maior_excursao <= pd.to_timedelta("15min"):
            return "ATENÇÃO", "🟡"
        return "CRÍTICO", "🔴"

    def sparkline(cur: pd.DataFrame) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cur["timestamp"], y=cur["value"], mode="lines",
            line=dict(width=1.6), hoverinfo="skip", showlegend=False
        ))
        fig.add_hline(y=anvisa_min, line_width=1, line_color="#e53935")
        fig.add_hline(y=anvisa_max, line_width=1, line_color="#fb8c00")
        fig.update_layout(
            margin=dict(l=0,r=0,t=0,b=0), height=80,
            xaxis=dict(visible=False), yaxis=dict(visible=False),
        )
        return fig

    # Tabela consolidada pra export/triagem rápida
    resumo_rows = []

    # Um “cartão” por sensor, fácil de ler
    for nome in df_all["sensor"].unique():
        cur = df_all[df_all["sensor"] == nome].copy()
        if cur.empty:
            continue

        # KPIs de qualidade e conformidade
        compl, exp, got, gap_total, gap_max = kpi_completude(cur, start_dt, end_dt, period)
        in_ct, out_ct, dur_in, dur_out, out_blocks, out_max = kpi_anvisa(cur, anvisa_min, anvisa_max)

        # Estatística básica da série
        t_min = cur["value"].min()
        t_p50 = cur["value"].median()
        t_max = cur["value"].max()
        std   = cur["value"].std(ddof=1)
        pct_fora = (out_ct / max(in_ct + out_ct, 1)) * 100.0

        status_txt, status_bullet = classifica_status(pct_fora, out_max)

        # Layout do cartão
        box = st.container(border=True)
        with box:
            top = st.columns([3,1.2,1.2,1.2])
            with top[0]:
                st.subheader(f"{status_bullet} {nome}")
                st.caption(f"Status: **{status_txt}** — {pct_fora:.1f}% fora • Maior excursão {fmt_td(out_max)}")
            with top[1]:
                st.metric("Completude", f"{compl:.2f}%", help=f"Esperado: {exp} • Recebido: {got}")
            with top[2]:
                st.metric("Tempo DENTRO", fmt_td(dur_in))
            with top[3]:
                st.metric("Tempo FORA", fmt_td(dur_out))

            # Barras de % dentro/fora
            dentro_pct = 100.0 - pct_fora
            b1, b2 = st.columns([3,2])
            with b1:
                st.progress(min(max(dentro_pct/100.0, 0), 1.0), text=f"Dentro da faixa: {dentro_pct:.1f}%")
            with b2:
                st.progress(min(max(pct_fora/100.0, 0), 1.0), text=f"Fora da faixa: {pct_fora:.1f}%")

            # Mini sparkline + estatísticas rápidas
            s1, s2, s3, s4 = st.columns([3,1,1,1])
            with s1:
                st.plotly_chart(sparkline(cur), use_container_width=True)
            with s2:
                st.metric("Mín (°C)", f"{t_min:.2f}")
                st.metric("P50 (°C)", f"{t_p50:.2f}")
            with s3:
                st.metric("Máx (°C)", f"{t_max:.2f}")
                st.metric("Desvio (°C)", f"{(0.0 if pd.isna(std) else std):.2f}")
            with s4:
                st.metric("Excursões", f"{out_blocks}")
                st.metric("Maior excursão", fmt_td(out_max))

            st.caption(f"Gaps totais: {fmt_td(gap_total)} • Maior gap: {fmt_td(gap_max)}")
        st.write("")  # espaçamento

        resumo_rows.append({
            "sensor": nome,
            "status": status_txt,
            "% fora": round(pct_fora, 2),
            "completude_%": compl,
            "tempo_dentro": fmt_td(dur_in),
            "tempo_fora": fmt_td(dur_out),
            "excursões": out_blocks,
            "maior_excursão": fmt_td(out_max),
            "mín_°C": round(t_min, 2),
            "p50_°C": round(t_p50, 2),
            "máx_°C": round(t_max, 2),
            "desvio_°C": round(0.0 if pd.isna(std) else float(std), 2),
            "gaps_total": fmt_td(gap_total),
            "maior_gap": fmt_td(gap_max),
        })

    st.markdown("### Tabela resumo (para triagem rápida)")
    if resumo_rows:
        df_resumo = pd.DataFrame(resumo_rows).sort_values(["status", "% fora", "sensor"])
        st.dataframe(df_resumo, use_container_width=True)
        st.download_button(
            "Baixar resumo (CSV)",
            df_resumo.to_csv(index=False).encode("utf-8"),
            file_name="kpis_resumo.csv",
            mime="text/csv",
        )
    else:
        st.info("Sem dados para resumir.")


    # ------------------------- SÉRIES -------------------------
    with tab_series:
        st.markdown("**Séries temporais com limites destacados** (verde dentro; vermelho/laranja fora).")

        if plot_all_together:
            # — MODO UNIFICADO (overlay)
            y_min_global = float(df_all["value"].min())
            y_max_global = float(df_all["value"].max())
            fig = go.Figure()

            # Bandas fortes da faixa ANVISA
            fig.add_hrect(y0=y_min_global - 1, y1=anvisa_min, fillcolor="#e53935", opacity=0.12, line_width=0, layer="below")
            fig.add_hrect(y0=anvisa_min, y1=anvisa_max, fillcolor="#2e7d32", opacity=0.18, line_width=0, layer="below")
            fig.add_hrect(y0=anvisa_max, y1=y_max_global + 1, fillcolor="#fb8c00", opacity=0.12, line_width=0, layer="below")
            fig.add_hline(y=anvisa_min, line_width=3, line_color="#e53935")
            fig.add_hline(y=anvisa_max, line_width=3, line_color="#fb8c00")

            for nome in df_all["sensor"].unique():
                cur = df_all[df_all["sensor"] == nome]
                if cur.empty:
                    continue
                fig.add_trace(go.Scatter(
                    x=cur["timestamp"], y=cur["value"], mode="lines", name=nome,
                    line=dict(width=2),
                    hovertemplate=f"{nome}<br>%{{x|%Y-%m-%d %H:%M}}<br>%{{y:.2f}} °C<extra></extra>"
                ))

            fig.update_layout(
                title="Série Temporal — Todos os dispositivos",
                xaxis_title="Tempo", yaxis_title="Temperatura (°C)",
                height=520, margin=dict(l=10, r=10, t=50, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig, use_container_width=True)

        else:
            # — MODO POR SENSOR (um gráfico por dispositivo)
            for nome in df_all["sensor"].unique():
                cur = df_all[df_all["sensor"] == nome]
                if cur.empty:
                    continue
                y_min = float(cur["value"].min())
                y_max = float(cur["value"].max())
                fig = go.Figure()
                fig.add_hrect(y0=y_min - 1, y1=anvisa_min, fillcolor="#e53935", opacity=0.12, line_width=0, layer="below")
                fig.add_hrect(y0=anvisa_min, y1=anvisa_max, fillcolor="#2e7d32", opacity=0.18, line_width=0, layer="below")
                fig.add_hrect(y0=anvisa_max, y1=y_max + 1, fillcolor="#fb8c00", opacity=0.12, line_width=0, layer="below")
                fig.add_hline(y=anvisa_min, line_width=3, line_color="#e53935")
                fig.add_hline(y=anvisa_max, line_width=3, line_color="#fb8c00")
                fig.add_trace(go.Scatter(
                    x=cur["timestamp"], y=cur["value"], mode="lines", name=nome,
                    line=dict(width=2),
                    hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.2f} °C<extra></extra>"
                ))
                # Pontos fora (deixa leve; se quiser, ative)
                below = cur[cur["value"] < anvisa_min]
                above = cur[cur["value"] > anvisa_max]
                if not below.empty:
                    fig.add_trace(go.Scatter(
                        x=below["timestamp"], y=below["value"], mode="markers",
                        name="Abaixo do mínimo", marker=dict(symbol="triangle-down", size=7, color="#e53935")))
                if not above.empty:
                    fig.add_trace(go.Scatter(
                        x=above["timestamp"], y=above["value"], mode="markers",
                        name="Acima do máximo", marker=dict(symbol="triangle-up", size=7, color="#fb8c00")))
                fig.update_layout(
                    title=f"Série Temporal — {nome}",
                    xaxis_title="Tempo", yaxis_title="Temperatura (°C)",
                    height=440, margin=dict(l=10, r=10, t=50, b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig, use_container_width=True)

    # ------------------------- HEATMAPS -------------------------
    with tab_heat:
        st.markdown("**Mapa de calor**: esquerda = temperatura média por Hora×Dia; direita = **% do tempo fora da faixa** (Hora×Dia).")
        for nome in df_all["sensor"].unique():
            cur = df_all[df_all["sensor"] == nome].copy()
            if cur.empty:
                continue
            cur["dia"] = cur["timestamp"].dt.date
            cur["hora"] = cur["timestamp"].dt.hour

            heat_mean = cur.groupby(["dia", "hora"])["value"].mean().reset_index()
            pv_mean = heat_mean.pivot(index="hora", columns="dia", values="value").sort_index()

            cur["fora"] = ~cur["value"].between(anvisa_min, anvisa_max)
            heat_out = cur.groupby(["dia", "hora"])["fora"].mean().reset_index()
            pv_out = (heat_out.pivot(index="hora", columns="dia", values="fora").sort_index() * 100.0)

            c1, c2 = st.columns(2)
            with c1:
                fig1 = px.imshow(
                    pv_mean, aspect="auto", origin="lower",
                    labels=dict(x="Dia", y="Hora", color="°C"),
                    title=f"{nome} — Temperatura média (Hora×Dia)",
                    color_continuous_scale="RdYlBu_r",
                )
                st.plotly_chart(fig1, use_container_width=True)

            with c2:
                fig2 = px.imshow(
                    pv_out, aspect="auto", origin="lower",
                    labels=dict(x="Dia", y="Hora", color="% fora"),
                    title=f"{nome} — % do tempo FORA da faixa (Hora×Dia)",
                    color_continuous_scale=["#2e7d32", "#fdd835", "#fb8c00", "#e53935"],
                )
                st.plotly_chart(fig2, use_container_width=True)

            st.caption("Quadrantes laranja/vermelho no % fora = períodos críticos.")

    # ------------------------- DISTRIBUIÇÃO -------------------------
    with tab_hist:
        st.markdown("**Distribuição**: histograma (densidade) + violin (dispersão), com linhas de limite.")
        for nome in df_all["sensor"].unique():
            cur = df_all[df_all["sensor"] == nome]
            if cur.empty:
                continue

            fig_h = px.histogram(
                cur, x="value", nbins=50, histnorm="probability density",
                title=f"Distribuição — {nome}",
            )
            fig_h.add_vline(x=anvisa_min, line_width=3, line_color="#e53935")
            fig_h.add_vline(x=anvisa_max, line_width=3, line_color="#fb8c00")
            fig_h.update_layout(
                xaxis_title="Temperatura (°C)", yaxis_title="Densidade",
                height=360, margin=dict(l=10, r=10, t=50, b=10),
            )

            fig_v = px.violin(cur, y="value", box=True, points="all", title=f"Violin — {nome}")
            fig_v.add_hline(y=anvisa_min, line_width=3, line_color="#e53935")
            fig_v.add_hline(y=anvisa_max, line_width=3, line_color="#fb8c00")
            fig_v.update_layout(
                yaxis_title="Temperatura (°C)", height=360, margin=dict(l=10, r=10, t=50, b=10),
            )

            c1, c2 = st.columns([2, 1])
            with c1:
                st.plotly_chart(fig_h, use_container_width=True)
            with c2:
                st.plotly_chart(fig_v, use_container_width=True)

            pct_out = (~cur["value"].between(anvisa_min, anvisa_max)).mean() * 100
            st.caption(f"≈ {pct_out:.1f}% fora da faixa ({anvisa_min:.1f}–{anvisa_max:.1f} °C).")

    # ------------------------- DADOS BRUTOS -------------------------
    with tab_raw:
        st.dataframe(
            df_all.sort_values(["sensor", "timestamp"]).reset_index(drop=True),
            use_container_width=True
        )
        st.download_button(
            "Baixar CSV consolidado",
            df_all.to_csv(index=False).encode("utf-8"),
            file_name="temperatura_consolidado.csv",
            mime="text/csv",
        )
