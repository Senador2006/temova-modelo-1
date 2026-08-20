"""Estado operacional compartilhado entre o dashboard e o app te-mova."""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ml.alert_levels import calcular_nivel_alerta

OPEN_STATUSES = {"enviada", "confirmada", "em_andamento", "interrompida"}
VALID_STATUSES = OPEN_STATUSES | {"concluida"}

STATUS_KIND = {
    "enviada": "envio",
    "confirmada": "confirmacao",
    "em_andamento": "andamento",
    "concluida": "finalizacao",
    "interrompida": "interrupcao",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _notification(dispatch_id: str, kind: str, message: str) -> dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "dispatchId": dispatch_id,
        "kind": kind,
        "message": message,
        "at": _now(),
    }


class OpsStore:
    """JSON em disco + lock, para sobreviver ao reload do uvicorn."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {
            "version": 0,
            "dispatches": [],
            "predictions": {},
            "field_updates": {},
            "app": None,
        }
        self._load()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._data))

    def set_presence(self, payload: dict[str, Any]) -> dict[str, Any]:
        nxt = {
            "url": payload.get("url") or "http://127.0.0.1:5050",
            "equipe": payload.get("equipe") or "Equipe Delta",
            "client": payload.get("client") or "app",
            "at": _now(),
        }
        with self._lock:
            prev = self._data.get("app") or {}
            self._data["app"] = nxt
            if prev.get("url") != nxt["url"] or prev.get("equipe") != nxt["equipe"]:
                self._bump_locked()
            else:
                self._save_locked()
            return json.loads(json.dumps(nxt))

    def create_dispatch(self, payload: dict[str, Any]) -> dict[str, Any]:
        dispatch_id = str(uuid.uuid4())
        created = _now()
        equipe = payload["equipe"]
        id_ponto = payload["idPonto"]
        road = payload.get("road") or id_ponto
        situacao = payload.get("situacao") or ""
        titulo = payload.get("titulo") or road

        dispatch = {
            "id": dispatch_id,
            "idPonto": id_ponto,
            "titulo": titulo,
            "road": road,
            "km": payload.get("km", 0),
            "local": payload.get("local") or "",
            "sentido": payload.get("sentido") or payload.get("local") or "Operação de campo",
            "equipe": equipe,
            "situacao": situacao,
            "nivelAlerta": payload.get("nivelAlerta") or "amarelo",
            "crescimentoCm": float(payload.get("crescimentoCm") or 0),
            "limitePodaCm": float(payload.get("limitePodaCm") or 10),
            "limiteAlertaCm": float(payload.get("limiteAlertaCm") or 9),
            "createdAt": created,
            "updatedAt": created,
            "status": "enviada",
            "alturaFinal": None,
            "foto": None,
            "notifications": [
                _notification(
                    dispatch_id,
                    "envio",
                    (
                        f"Tarefa enviada ao app de {equipe} para {road} ({id_ponto}). "
                        f"Situação: {situacao}"
                    ),
                )
            ],
        }

        with self._lock:
            self._data["dispatches"].insert(0, dispatch)
            self._bump_locked()
            return json.loads(json.dumps(dispatch))

    def patch_dispatch(self, dispatch_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            dispatch = next(
                (item for item in self._data["dispatches"] if item["id"] == dispatch_id),
                None,
            )
            if dispatch is None:
                raise KeyError(dispatch_id)

            next_status = payload.get("status")
            if next_status is not None:
                if next_status not in VALID_STATUSES:
                    raise ValueError(f"Status inválido: {next_status}")
                if next_status != dispatch["status"]:
                    dispatch["status"] = next_status
                    dispatch["notifications"].insert(
                        0,
                        _notification(
                            dispatch_id,
                            STATUS_KIND[next_status],
                            self._status_message(dispatch, next_status, payload),
                        ),
                    )

            if payload.get("alturaFinal") is not None:
                dispatch["alturaFinal"] = float(payload["alturaFinal"])
            if payload.get("foto") is not None:
                dispatch["foto"] = payload["foto"]

            dispatch["updatedAt"] = _now()

            if dispatch["status"] == "concluida" and dispatch.get("alturaFinal") is not None:
                self._apply_field_update_locked(dispatch)

            self._bump_locked()
            return json.loads(json.dumps(dispatch))

    def set_predictions(self, predictions: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            self._data["predictions"] = predictions
            for dispatch in self._data["dispatches"]:
                if dispatch["status"] not in OPEN_STATUSES:
                    continue
                pred = predictions.get(dispatch["idPonto"])
                if not isinstance(pred, dict):
                    continue
                if pred.get("crescimento_acumulado_cm") is not None:
                    dispatch["crescimentoCm"] = float(pred["crescimento_acumulado_cm"])
                if pred.get("nivel_alerta"):
                    dispatch["nivelAlerta"] = pred["nivel_alerta"]
                if pred.get("limite_poda_cm") is not None:
                    dispatch["limitePodaCm"] = float(pred["limite_poda_cm"])
                if pred.get("limite_alerta_cm") is not None:
                    dispatch["limiteAlertaCm"] = float(pred["limite_alerta_cm"])
                dispatch["updatedAt"] = _now()
            self._bump_locked()
            return json.loads(json.dumps(self._data))

    def _apply_field_update_locked(self, dispatch: dict[str, Any]) -> None:
        altura = float(dispatch["alturaFinal"])
        alerta = calcular_nivel_alerta(altura)
        self._data["field_updates"][dispatch["idPonto"]] = {
            "idPonto": dispatch["idPonto"],
            "alturaCm": altura,
            "nivelAlerta": alerta["nivel_alerta"],
            "nivelLabel": alerta["nivel_label"],
            "limitePodaCm": alerta["limite_poda_cm"],
            "at": _now(),
            "dispatchId": dispatch["id"],
            "equipe": dispatch["equipe"],
            "fonte": "app",
            "foto": dispatch.get("foto"),
        }

    def _status_message(
        self,
        dispatch: dict[str, Any],
        status: str,
        payload: dict[str, Any],
    ) -> str:
        equipe = dispatch["equipe"]
        ponto = dispatch["idPonto"]
        road = dispatch.get("road") or ponto
        source = payload.get("source") or "app"
        origem = "no app" if source == "app" else "pelo dashboard"

        messages = {
            "enviada": f"Tarefa reenviada a {equipe}.",
            "confirmada": f"{equipe} confirmou recebimento {origem} para {road} ({ponto}).",
            "em_andamento": f"{equipe} iniciou a poda em {ponto} ({dispatch.get('local') or 'trecho'}).",
            "concluida": (
                f"{equipe} finalizou a poda em {ponto}. "
                + (
                    f"Altura medida: {float(payload.get('alturaFinal') or dispatch.get('alturaFinal') or 0):.1f} cm. "
                    if payload.get("alturaFinal") is not None or dispatch.get("alturaFinal") is not None
                    else "Trecho liberado. "
                )
            ),
            "interrompida": f"{equipe} interrompeu o atendimento em {ponto}. Chamado segue aberto.",
        }
        return messages[status]

    def _bump_locked(self) -> None:
        self._data["version"] = int(self._data.get("version") or 0) + 1
        self._save_locked()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if isinstance(loaded, dict):
            self._data.update(loaded)
            self._data.setdefault("version", 0)
            self._data.setdefault("dispatches", [])
            self._data.setdefault("predictions", {})
            self._data.setdefault("field_updates", {})
            self._data.setdefault("app", None)

    def _save_locked(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(self.path)
