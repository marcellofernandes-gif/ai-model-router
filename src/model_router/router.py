"""Roteador que escolhe o modelo Claude certo para cada função.

Uso básico:

    from model_router import Roteador

    r = Roteador()
    res = r.executar("resumir", "Texto longo aqui...")
    print(res.texto, res.modelo, res.custo_usd)
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Callable

import anthropic

# Fallback automático no servidor: se o modelo recusar por política de
# segurança, a própria API refaz o pedido em outro modelo adequado.
# Disponível em Opus 5.5 e Sonnet 5.5 (não existe para Haiku).
FALLBACK_BETA = "server-side-fallback-2026-07-01"
MODELOS_COM_FALLBACK_SERVIDOR = {"claude-opus-5-5", "claude-sonnet-5-5"}

Validador = Callable[[str], bool]


# --------------------------------------------------------------------------- config


@dataclass(frozen=True)
class Modelo:
    apelido: str
    id: str
    preco_entrada: float
    preco_saida: float


@dataclass(frozen=True)
class Funcao:
    nome: str
    descricao: str
    modelo: str
    effort: str
    max_tokens: int
    escalar_para: str | None = None


@dataclass(frozen=True)
class Config:
    modelos: dict[str, Modelo]
    funcoes: dict[str, Funcao]

    def modelo_por_id(self, model_id: str) -> Modelo | None:
        return next((m for m in self.modelos.values() if m.id == model_id), None)


def carregar_config(caminho: str | Path | None = None) -> Config:
    """Lê o arquivo de rotas. Sem caminho, usa o `rotas.toml` do pacote."""
    if caminho is None:
        texto = resources.files("model_router").joinpath("rotas.toml").read_text("utf-8")
    else:
        texto = Path(caminho).read_text("utf-8")
    dados = tomllib.loads(texto)

    modelos = {
        apelido: Modelo(apelido, m["id"], float(m["preco_entrada"]), float(m["preco_saida"]))
        for apelido, m in dados["modelos"].items()
    }
    funcoes: dict[str, Funcao] = {}
    for nome, f in dados["funcoes"].items():
        funcao = Funcao(
            nome=nome,
            descricao=f["descricao"],
            modelo=f["modelo"],
            effort=f.get("effort", "medium"),
            max_tokens=int(f.get("max_tokens", 16000)),
            escalar_para=f.get("escalar_para"),
        )
        for apelido in filter(None, [funcao.modelo, funcao.escalar_para]):
            if apelido not in modelos:
                raise ValueError(f"Função '{nome}' usa o modelo '{apelido}', que não está em [modelos].")
        funcoes[nome] = funcao
    return Config(modelos, funcoes)


# --------------------------------------------------------------------------- resultado


@dataclass
class Tentativa:
    modelo: str
    motivo_falha: str | None  # None = esta tentativa foi aceita


@dataclass
class Resultado:
    texto: str
    funcao: str
    modelo: str  # modelo que efetivamente respondeu
    stop_reason: str | None
    tokens_entrada: int
    tokens_saida: int
    custo_usd: float | None
    escalou: bool = False
    tentativas: list[Tentativa] = field(default_factory=list)

    @property
    def truncado(self) -> bool:
        return self.stop_reason == "max_tokens"


class SemRespostaValida(RuntimeError):
    """Nenhum modelo da cadeia entregou uma resposta aceitável."""

    def __init__(self, funcao: str, tentativas: list[Tentativa]):
        self.tentativas = tentativas
        detalhes = "; ".join(f"{t.modelo}: {t.motivo_falha}" for t in tentativas)
        super().__init__(f"Função '{funcao}' sem resposta válida ({detalhes})")


# --------------------------------------------------------------------------- roteador


class Roteador:
    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        config: Config | None = None,
    ) -> None:
        self.client = client or anthropic.Anthropic()
        self.config = config or carregar_config()

    # ------------------------------------------------------------------ público

    def funcoes(self) -> dict[str, Funcao]:
        return dict(self.config.funcoes)

    def executar(
        self,
        funcao: str,
        prompt: str,
        *,
        system: str | None = None,
        validar: Validador | None = None,
    ) -> Resultado:
        """Executa `prompt` com o modelo definido para `funcao`.

        Sobe para o próximo modelo da cadeia (`escalar_para`) quando a resposta
        é uma recusa ou quando `validar(texto)` retorna False.
        """
        if funcao not in self.config.funcoes:
            opcoes = ", ".join(sorted(self.config.funcoes))
            raise KeyError(f"Função desconhecida: '{funcao}'. Opções: {opcoes}")
        f = self.config.funcoes[funcao]

        tentativas: list[Tentativa] = []
        for apelido in self._cadeia(f):
            modelo = self.config.modelos[apelido]
            resposta = self._chamar(modelo, f, prompt, system)
            texto = "".join(b.text for b in resposta.content if b.type == "text")

            motivo = self._motivo_rejeicao(resposta, texto, validar)
            tentativas.append(Tentativa(resposta.model, motivo))
            if motivo is None:
                return self._resultado(f, resposta, texto, tentativas)

        raise SemRespostaValida(funcao, tentativas)

    def auto(
        self,
        prompt: str,
        *,
        system: str | None = None,
        validar: Validador | None = None,
    ) -> Resultado:
        """Descobre a função certa para o pedido (com Haiku) e executa."""
        funcao, confianca = self.escolher_funcao(prompt)
        if confianca == "baixa":
            # Na dúvida, não arrisca qualidade: usa a função de raciocínio.
            funcao = "analisar" if "analisar" in self.config.funcoes else funcao
        return self.executar(funcao, prompt, system=system, validar=validar)

    def escolher_funcao(self, prompt: str) -> tuple[str, str]:
        """Classifica o pedido em uma das funções configuradas."""
        nomes = sorted(self.config.funcoes)
        catalogo = "\n".join(f"- {n}: {self.config.funcoes[n].descricao}" for n in nomes)
        classificador = self.config.modelos.get("haiku") or next(iter(self.config.modelos.values()))

        resposta = self.client.messages.create(
            model=classificador.id,
            max_tokens=1024,
            output_config={
                "effort": "low",
                "format": {
                    "type": "json_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "funcao": {"type": "string", "enum": nomes},
                            "confianca": {"type": "string", "enum": ["alta", "baixa"]},
                        },
                        "required": ["funcao", "confianca"],
                        "additionalProperties": False,
                    },
                },
            },
            system=(
                "Você escolhe qual função atende melhor o pedido do usuário.\n"
                f"Funções disponíveis:\n{catalogo}\n\n"
                "Use confianca='baixa' se o pedido for ambíguo, misturar várias "
                "funções ou parecer exigir raciocínio mais profundo do que a "
                "descrição da função sugere."
            ),
            messages=[{"role": "user", "content": f"<pedido>\n{prompt}\n</pedido>"}],
        )
        if resposta.stop_reason == "refusal":
            return ("analisar" if "analisar" in self.config.funcoes else nomes[0]), "baixa"
        texto = next(b.text for b in resposta.content if b.type == "text")
        dados = json.loads(texto)
        return dados["funcao"], dados["confianca"]

    # ------------------------------------------------------------------ interno

    def _cadeia(self, f: Funcao) -> list[str]:
        """Modelos a tentar, em ordem. Ex.: haiku -> sonnet -> opus.

        Depois do primeiro degrau (`escalar_para` da própria função), continua
        subindo pelo `escalar_para` de outras funções que começam no modelo atual.
        """
        cadeia = [f.modelo]
        proximo = f.escalar_para
        while proximo and proximo not in cadeia:
            cadeia.append(proximo)
            proximo = self._escalada_padrao(proximo)
        return cadeia

    def _escalada_padrao(self, apelido: str) -> str | None:
        for outra in self.config.funcoes.values():
            if outra.modelo == apelido and outra.escalar_para:
                return outra.escalar_para
        return None

    def _chamar(self, modelo: Modelo, f: Funcao, prompt: str, system: str | None) -> Any:
        params: dict[str, Any] = {
            "model": modelo.id,
            "max_tokens": f.max_tokens,
            "output_config": {"effort": f.effort},
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            params["system"] = system

        if modelo.id in MODELOS_COM_FALLBACK_SERVIDOR:
            return self.client.beta.messages.create(
                betas=[FALLBACK_BETA], fallbacks="default", **params
            )
        return self.client.messages.create(**params)

    @staticmethod
    def _motivo_rejeicao(resposta: Any, texto: str, validar: Validador | None) -> str | None:
        if resposta.stop_reason == "refusal":
            return "recusa"
        if not texto.strip():
            return "resposta vazia"
        if validar is not None and resposta.stop_reason != "max_tokens":
            try:
                if not validar(texto):
                    return "não passou na validação"
            except Exception as e:  # validador com erro conta como reprovado
                return f"erro na validação: {e}"
        return None

    def _resultado(self, f: Funcao, resposta: Any, texto: str, tentativas: list[Tentativa]) -> Resultado:
        uso = resposta.usage
        entrada = (
            (uso.input_tokens or 0)
            + (getattr(uso, "cache_creation_input_tokens", 0) or 0)
            + (getattr(uso, "cache_read_input_tokens", 0) or 0)
        )
        saida = uso.output_tokens or 0
        modelo = self.config.modelo_por_id(resposta.model)
        custo = None
        if modelo:
            custo = round((entrada * modelo.preco_entrada + saida * modelo.preco_saida) / 1_000_000, 6)
        return Resultado(
            texto=texto,
            funcao=f.nome,
            modelo=resposta.model,
            stop_reason=resposta.stop_reason,
            tokens_entrada=entrada,
            tokens_saida=saida,
            custo_usd=custo,
            escalou=len(tentativas) > 1,
            tentativas=tentativas,
        )
