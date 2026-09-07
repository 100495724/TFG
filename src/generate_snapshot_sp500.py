#!/usr/bin/env python3
"""
snapshot_sp500.py
Captura un snapshot estatico de datos de companias del S&P 500 desde yfinance.

Por cada ticker se descarga:
  - .info  (dict completo con fundamentales, sector, pais, CEO, etc.)
  - .history(period='6mo')  (precios diarios OHLCV ultimos 6 meses)
  - .dividends  (historial completo de dividendos)
  - .news  (ultimas N noticias)

Salidas (por defecto en data/snapshots/<YYYY-MM-DD>/):
  - companies/<TICKER>.json   un fichero por ticker (permite reanudar)
  - sp500_snapshot_<DATE>.json   fichero agregado final con metadatos
  - snapshot.log   log completo de la ejecucion

Uso:
  python snapshot_sp500.py                  # snapshot completo
  python snapshot_sp500.py --limit 10       # solo 10 tickers (testing)
  python snapshot_sp500.py --ticker AAPL --ticker MSFT
  python snapshot_sp500.py --resume         # saltarse los ya descargados

Dependencias: yfinance, pandas, lxml (para pd.read_html sobre Wikipedia).
  pip install yfinance pandas lxml tqdm
"""

import argparse
import json
import logging
import sys
import time
import requests
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


# ============================================================
# Configuration
# ============================================================
DEFAULT_OUTPUT_DIR = Path("data/snapshots")
SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SLEEP_BETWEEN_TICKERS = 0.5    # segundos entre requests
MAX_RETRIES = 3
RETRY_BACKOFF = 2.0            # se multiplica por el numero de intento
HISTORY_PERIOD = "6mo"
NEWS_MAX_ITEMS = 5


# ============================================================
# Helpers
# ============================================================

def setup_logging(output_dir: Path) -> None:
    """Configura logging a stdout y a fichero."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "snapshot.log"
    # Si ya hay handlers (re-llamada en tests), los limpiamos
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def json_safe(obj):
    """Conversor para tipos no serializables por json.dump (numpy, datetime, etc)."""
    if hasattr(obj, "item"):       # numpy scalar
        return obj.item()
    if hasattr(obj, "isoformat"):  # datetime / Timestamp
        return obj.isoformat()
    return str(obj)


def fetch_sp500_constituents() -> list[dict]:
    """Descarga la lista actual de constituyentes del S&P 500 desde Wikipedia."""
    logging.info("Descargando lista S&P 500 desde Wikipedia...")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    # Hacemos la petición con nuestras cabeceras falsas
    respuesta = requests.get(SP500_WIKI_URL, headers=headers)
    
    # Ahora sí le pasamos el texto HTML a pandas
    tables = pd.read_html(respuesta.text)
    df = tables[0]
    constituents = []
    for _, row in df.iterrows():
        # yfinance usa BRK-B en vez de BRK.B
        ticker_yf = str(row["Symbol"]).replace(".", "-")
        constituents.append({
            "ticker": ticker_yf,
            "name": row["Security"],
            "sector": row.get("GICS Sector", ""),
            "sub_industry": row.get("GICS Sub-Industry", ""),
            "headquarters": row.get("Headquarters Location", ""),
        })
    logging.info(f"Encontrados {len(constituents)} constituyentes.")
    return constituents


def normalize_news_item(item: dict) -> dict:
    """Normaliza el formato de noticias (cambia entre versiones de yfinance)."""
    if "content" in item:
        # Formato nuevo (yfinance 0.2.x+): todo bajo 'content'
        content = item.get("content") or {}
        provider = content.get("provider") or {}
        canonical = content.get("canonicalUrl") or {}
        return {
            "title": content.get("title", ""),
            "summary": content.get("summary") or content.get("description", ""),
            "publisher": provider.get("displayName", ""),
            "publish_time": content.get("pubDate", ""),
            "link": canonical.get("url", ""),
        }
    # Formato antiguo
    return {
        "title": item.get("title", ""),
        "summary": item.get("summary", ""),
        "publisher": item.get("publisher", ""),
        "publish_time": item.get("providerPublishTime", ""),
        "link": item.get("link", ""),
    }


def is_info_valid(info: dict) -> bool:
    """Comprueba si .info contiene datos reales y no un dict casi vacio."""
    if not info:
        return False
    # yfinance a veces devuelve {'trailingPegRatio': None} para tickers invalidos
    meaningful = [k for k in info if info[k] not in (None, "", [], {})]
    return len(meaningful) >= 5


def fetch_ticker_data(ticker: str) -> dict:
    """Descarga TODOS los datos para un ticker. Lanza excepcion si .info no es valido."""
    yf_ticker = yf.Ticker(ticker)
    data = {
        "ticker": ticker,
        "fetched_at": datetime.utcnow().isoformat() + "Z",
    }

    # --- info (critico) ---
    info = yf_ticker.info
    if not is_info_valid(info):
        raise ValueError(f".info vacio o invalido (claves: {list(info.keys())[:5]})")
    data["info"] = info

    # --- history (best-effort) ---
    try:
        hist = yf_ticker.history(period=HISTORY_PERIOD, auto_adjust=True)
        if not hist.empty:
            hist = hist.reset_index()
            hist["Date"] = pd.to_datetime(hist["Date"]).dt.strftime("%Y-%m-%d")
            data["history"] = hist.to_dict(orient="records")
        else:
            data["history"] = []
    except Exception as e:
        logging.warning(f"{ticker}: history fallo ({e})")
        data["history"] = []

    # --- dividends (best-effort) ---
    try:
        divs = yf_ticker.dividends
        if divs is not None and not divs.empty:
            data["dividends"] = [
                {"date": d.strftime("%Y-%m-%d"), "amount": float(amount)}
                for d, amount in divs.items()
            ]
        else:
            data["dividends"] = []
    except Exception as e:
        logging.warning(f"{ticker}: dividends fallo ({e})")
        data["dividends"] = []

    # --- news (best-effort) ---
    try:
        news_raw = yf_ticker.news or []
        data["news"] = [normalize_news_item(item) for item in news_raw[:NEWS_MAX_ITEMS]]
    except Exception as e:
        logging.warning(f"{ticker}: news fallo ({e})")
        data["news"] = []

    return data


def fetch_with_retries(ticker: str) -> dict:
    """Envuelve fetch_ticker_data con reintentos y backoff exponencial."""
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fetch_ticker_data(ticker)
        except Exception as e:
            last_error = e
            logging.warning(f"{ticker}: intento {attempt}/{MAX_RETRIES} fallo: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)
    raise last_error


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Snapshot estatico del S&P 500 via yfinance.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help=f"Directorio raiz de salida (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--limit", type=int, default=None,
                        help="Procesar solo los primeros N tickers (testing)")
    parser.add_argument("--resume", action="store_true",
                        help="Saltarse tickers que ya tienen JSON en companies/")
    parser.add_argument("--ticker", action="append",
                        help="Procesar solo este ticker (se puede repetir)")
    args = parser.parse_args()

    today = datetime.utcnow().strftime("%Y-%m-%d")
    output_dir = args.output_dir / today
    companies_dir = output_dir / "companies"
    companies_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)

    logging.info("=" * 60)
    logging.info(f"S&P 500 SNAPSHOT  —  {today}")
    logging.info(f"yfinance: {yf.__version__}")
    logging.info("=" * 60)

    # ---------- 1) Universo ----------
    constituents = fetch_sp500_constituents()
    if args.ticker:
        wanted = set(args.ticker)
        constituents = [c for c in constituents if c["ticker"] in wanted]
        logging.info(f"Filtrando a {len(constituents)} tickers especificados.")
    if args.limit:
        constituents = constituents[: args.limit]
        logging.info(f"Limitando a los primeros {args.limit} tickers.")

    # ---------- 2) Descarga por ticker ----------
    failures: list[dict] = []
    n_processed = 0
    n_skipped = 0

    iterator = tqdm(constituents, desc="Tickers", unit="t")
    for entry in iterator:
        ticker = entry["ticker"]
        ticker_path = companies_dir / f"{ticker}.json"

        if args.resume and ticker_path.exists():
            n_skipped += 1
            continue

        try:
            data = fetch_with_retries(ticker)
            data["wikipedia_meta"] = entry
            with open(ticker_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=json_safe)
            n_processed += 1
        except Exception as e:
            logging.error(f"{ticker}: ABANDONADO tras {MAX_RETRIES} intentos: {e}")
            failures.append({"ticker": ticker, "reason": str(e)})

        time.sleep(SLEEP_BETWEEN_TICKERS)

    # ---------- 3) Agregar a un solo fichero ----------
    logging.info("Agregando ficheros individuales en snapshot final...")
    companies: dict[str, dict] = {}
    for ticker_file in sorted(companies_dir.glob("*.json")):
        try:
            with open(ticker_file, encoding="utf-8") as f:
                td = json.load(f)
            companies[td["ticker"]] = td
        except Exception as e:
            logging.warning(f"No se pudo cargar {ticker_file.name}: {e}")

    snapshot = {
        "snapshot_metadata": {
            "date": today,
            "fetched_at": datetime.utcnow().isoformat() + "Z",
            "yfinance_version": yf.__version__,
            "source_ticker_list": "Wikipedia S&P 500 constituents",
            "history_period": HISTORY_PERIOD,
            "news_max_items": NEWS_MAX_ITEMS,
            "n_tickers_in_universe": len(constituents) + n_skipped,
            "n_tickers_successful": len(companies),
            "n_failures": len(failures),
            "failures": failures,
        },
        "companies": companies,
    }

    final_path = output_dir / f"sp500_snapshot_{today}.json"
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False, default=json_safe)

    # ---------- 4) Resumen ----------
    logging.info("=" * 60)
    logging.info("SNAPSHOT COMPLETADO")
    logging.info("=" * 60)
    logging.info(f"  Tickers procesados esta ejecucion : {n_processed}")
    logging.info(f"  Tickers saltados (--resume)       : {n_skipped}")
    logging.info(f"  Fallos                            : {len(failures)}")
    logging.info(f"  Total en snapshot                 : {len(companies)}")
    logging.info(f"  Ficheros por ticker               : {companies_dir}")
    logging.info(f"  Snapshot agregado                 : {final_path}")
    if failures:
        logging.info("  Fallos detallados:")
        for fail in failures:
            logging.info(f"    - {fail['ticker']}: {fail['reason']}")


if __name__ == "__main__":
    main()