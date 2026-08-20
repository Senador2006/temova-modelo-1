from ml.config import LIMITE_ALERTA_CM, LIMITE_PODA_CM

NIVEIS = {
    "verde": {
        "label": "Normal",
        "descricao": "Crescimento dentro da faixa segura. Poda não necessária.",
    },
    "amarelo": {
        "label": "Atenção",
        "descricao": "Grama se aproximando do limite. Monitorar de perto (alerta preventivo a partir de 6 cm).",
    },
    "vermelho": {
        "label": "Alerta preventivo",
        "descricao": f"Zona de alerta ({LIMITE_ALERTA_CM}–{LIMITE_PODA_CM} cm) — intervenção iminente antes da poda.",
    },
    "laranja": {
        "label": "Poda necessária",
        "descricao": f"Grama atingiu ou ultrapassou o limite operacional de {LIMITE_PODA_CM} cm. Poda recomendada.",
    },
}


def calcular_nivel_alerta(
    valor_cm: float,
    limite_poda: float = LIMITE_PODA_CM,
    limite_alerta: float = LIMITE_ALERTA_CM,
) -> dict:
    """Classifica a previsão: alerta preventivo em 9 cm, poda em 10 cm."""
    if valor_cm >= limite_poda:
        nivel = "laranja"
    elif valor_cm >= limite_alerta:
        nivel = "vermelho"
    elif valor_cm >= 6.0:
        nivel = "amarelo"
    else:
        nivel = "verde"

    info = NIVEIS[nivel]
    distancia_poda = round(abs(valor_cm - limite_poda), 2)
    distancia_alerta = round(abs(valor_cm - limite_alerta), 2)

    return {
        "nivel_alerta": nivel,
        "nivel_label": info["label"],
        "mensagem_alerta": info["descricao"],
        "limite_alerta_cm": limite_alerta,
        "limite_poda_cm": limite_poda,
        "distancia_limite_cm": distancia_poda,
        "distancia_alerta_cm": distancia_alerta,
        "alerta_preventivo": valor_cm >= limite_alerta,
        "proximo_limite": valor_cm >= 6.0,
    }
