"""Testes sem chamar a API: usam um cliente falso."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from model_router import Roteador, SemRespostaValida, carregar_config
from model_router.cli import main


def resposta(model: str, texto: str = "ok", stop_reason: str = "end_turn"):
    return SimpleNamespace(
        model=model,
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=texto)],
        usage=SimpleNamespace(
            input_tokens=1000, output_tokens=500,
            cache_creation_input_tokens=0, cache_read_input_tokens=0,
        ),
    )


class ClienteFalso:
    """Devolve respostas em fila e guarda os parâmetros de cada chamada."""

    def __init__(self, respostas):
        self.fila = list(respostas)
        self.chamadas: list[tuple[str, dict]] = []
        self.messages = SimpleNamespace(create=lambda **kw: self._criar("messages", kw))
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: self._criar("beta", kw)))

    def _criar(self, rota, kw):
        self.chamadas.append((rota, kw))
        r = self.fila.pop(0)
        return r(kw["model"]) if callable(r) else r


def test_config_padrao_carrega_e_e_coerente():
    cfg = carregar_config()
    assert {"haiku", "sonnet", "opus"} <= set(cfg.modelos)
    assert cfg.funcoes["classificar"].modelo == "haiku"
    assert cfg.funcoes["resumir"].modelo == "sonnet"
    assert cfg.funcoes["codigo"].modelo == "opus"


def test_haiku_usa_endpoint_normal_sem_fallback():
    cli = ClienteFalso([lambda m: resposta(m, "positivo")])
    res = Roteador(cli).executar("classificar", "Adorei o produto")
    rota, kw = cli.chamadas[0]
    assert rota == "messages"
    assert kw["model"] == "claude-haiku-5-5"
    assert kw["output_config"] == {"effort": "low"}
    assert "fallbacks" not in kw and "thinking" not in kw
    assert res.texto == "positivo" and not res.escalou


def test_opus_e_sonnet_usam_fallback_do_servidor():
    cli = ClienteFalso([lambda m: resposta(m), lambda m: resposta(m)])
    r = Roteador(cli)
    r.executar("codigo", "faça X")
    r.executar("resumir", "texto")
    for rota, kw in cli.chamadas:
        assert rota == "beta"
        assert kw["fallbacks"] == "default"
        assert kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert cli.chamadas[0][1]["model"] == "claude-opus-5-5"
    assert cli.chamadas[0][1]["output_config"] == {"effort": "high"}


def test_recusa_no_haiku_escala_para_sonnet():
    cli = ClienteFalso([
        lambda m: resposta(m, "", stop_reason="refusal"),
        lambda m: resposta(m, "dados"),
    ])
    res = Roteador(cli).executar("extrair", "nota fiscal...")
    assert [kw["model"] for _, kw in cli.chamadas] == ["claude-haiku-5-5", "claude-sonnet-5-5"]
    assert res.escalou and res.modelo == "claude-sonnet-5-5"
    assert res.tentativas[0].motivo_falha == "recusa"


def test_validacao_reprovada_sobe_ate_opus():
    def json_valido(t):
        json.loads(t)
        return True

    cli = ClienteFalso([
        lambda m: resposta(m, "não é json"),
        lambda m: resposta(m, "ainda não"),
        lambda m: resposta(m, '{"total": 10}'),
    ])
    res = Roteador(cli).executar("extrair", "...", validar=json_valido)
    assert [kw["model"] for _, kw in cli.chamadas] == [
        "claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5",
    ]
    assert res.texto == '{"total": 10}'


def test_cadeia_esgotada_levanta_erro():
    cli = ClienteFalso([lambda m: resposta(m, "", stop_reason="refusal")])
    with pytest.raises(SemRespostaValida):
        Roteador(cli).executar("codigo", "...")


def test_custo_calculado_pelo_modelo_que_respondeu():
    cli = ClienteFalso([lambda m: resposta(m)])
    res = Roteador(cli).executar("analisar", "...")
    # 1000 * 4.00/1M + 500 * 20.00/1M
    assert res.custo_usd == pytest.approx(0.014)


def test_funcao_desconhecida():
    with pytest.raises(KeyError):
        Roteador(ClienteFalso([])).executar("inexistente", "x")


def test_auto_escolhe_funcao_e_executa():
    cli = ClienteFalso([
        lambda m: resposta(m, json.dumps({"funcao": "traduzir", "confianca": "alta"})),
        lambda m: resposta(m, "Hello"),
    ])
    res = Roteador(cli).auto("Traduza 'olá' para inglês")
    assert cli.chamadas[0][1]["model"] == "claude-haiku-5-5"
    assert res.funcao == "traduzir" and res.modelo == "claude-sonnet-5-5"


def test_auto_com_baixa_confianca_usa_raciocinio():
    cli = ClienteFalso([
        lambda m: resposta(m, json.dumps({"funcao": "resumir", "confianca": "baixa"})),
        lambda m: resposta(m, "análise"),
    ])
    res = Roteador(cli).auto("pedido ambíguo")
    assert res.funcao == "analisar" and res.modelo == "claude-opus-5-5"


def test_config_com_modelo_inexistente(tmp_path):
    p = tmp_path / "rotas.toml"
    p.write_text(
        '[modelos.a]\nid="x"\npreco_entrada=1\npreco_saida=1\n'
        '[funcoes.f]\ndescricao="d"\nmodelo="b"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        carregar_config(p)


def test_cli_lista_funcoes(capsys):
    assert main(["funcoes"]) == 0
    saida = capsys.readouterr().out
    assert "classificar" in saida and "haiku -> sonnet" in saida
