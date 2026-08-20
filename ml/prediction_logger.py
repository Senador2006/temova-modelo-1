"""Persistência e consulta de logs de predição para monitoramento e dashboards."""
from __future__ import annotations

import json
import threading
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


def _parse_timestamp(value: str) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


class PredictionLogger:
    """Append-only JSONL logger para predições do modelo."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(self, entry: dict[str, Any]) -> str:
        prediction_id = str(uuid.uuid4())
        record = {
            "prediction_id": prediction_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **entry,
        }
        line = json.dumps(record, ensure_ascii=False)

        with self._lock:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

        return prediction_id

    def _iter_entries(self) -> list[dict[str, Any]]:
        if not self.log_path.exists():
            return []

        entries: list[dict[str, Any]] = []
        with self.log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return entries

    def query(
        self,
        limit: int = 100,
        id_ponto: str | None = None,
        nivel_alerta: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> list[dict[str, Any]]:
        filtered: list[dict[str, Any]] = []

        for entry in self._iter_entries():
            if id_ponto and entry.get("id_ponto") != id_ponto:
                continue
            if nivel_alerta and entry.get("nivel_alerta") != nivel_alerta:
                continue

            ts = _parse_timestamp(entry.get("timestamp", ""))
            if since and (ts is None or ts < since):
                continue
            if until and (ts is None or ts > until):
                continue

            filtered.append(entry)

        filtered.sort(key=lambda item: item.get("timestamp", ""), reverse=True)
        return filtered[:limit]

    def stats(self, window_hours: int = 24) -> dict[str, Any]:
        entries = self._iter_entries()
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(hours=window_hours)

        window_entries: list[dict[str, Any]] = []
        all_by_nivel: Counter[str] = Counter()
        window_by_nivel: Counter[str] = Counter()
        by_ponto: dict[str, dict[str, Any]] = {}

        crescimentos: list[float] = []
        alertas_preventivos = 0
        podas_necessarias = 0

        for entry in entries:
            nivel = entry.get("nivel_alerta", "desconhecido")
            all_by_nivel[nivel] += 1

            ts = _parse_timestamp(entry.get("timestamp", ""))
            in_window = ts is not None and ts >= window_start
            if in_window:
                window_entries.append(entry)
                window_by_nivel[nivel] += 1

                crescimento = entry.get("crescimento_acumulado_cm")
                if isinstance(crescimento, (int, float)):
                    crescimentos.append(float(crescimento))
                if entry.get("alerta_preventivo"):
                    alertas_preventivos += 1
                if entry.get("acima_limite_poda"):
                    podas_necessarias += 1

            ponto = entry.get("id_ponto")
            if ponto and (ponto not in by_ponto or entry.get("timestamp", "") > by_ponto[ponto].get("timestamp", "")):
                by_ponto[ponto] = {
                    "id_ponto": ponto,
                    "timestamp": entry.get("timestamp"),
                    "data_referencia": entry.get("data_referencia"),
                    "crescimento_acumulado_cm": entry.get("crescimento_acumulado_cm"),
                    "nivel_alerta": entry.get("nivel_alerta"),
                    "nivel_label": entry.get("nivel_label"),
                    "alerta_preventivo": entry.get("alerta_preventivo"),
                    "acima_limite_poda": entry.get("acima_limite_poda"),
                }

        total_window = len(window_entries)
        avg_growth = round(sum(crescimentos) / len(crescimentos), 2) if crescimentos else None

        return {
            "total_predictions": len(entries),
            "window_hours": window_hours,
            "predictions_in_window": total_window,
            "avg_crescimento_cm": avg_growth,
            "alerta_preventivo_count": alertas_preventivos,
            "poda_necessaria_count": podas_necessarias,
            "alerta_preventivo_rate_pct": round(100 * alertas_preventivos / total_window, 1) if total_window else 0.0,
            "poda_necessaria_rate_pct": round(100 * podas_necessarias / total_window, 1) if total_window else 0.0,
            "by_nivel_alerta_all_time": dict(all_by_nivel),
            "by_nivel_alerta_window": dict(window_by_nivel),
            "latest_by_ponto": sorted(by_ponto.values(), key=lambda item: item.get("id_ponto", "")),
            "ultima_predicao": entries[-1] if entries else None,
        }
