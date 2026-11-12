"""Linha de comando.

    ia funcoes
    ia resumir "texto..."
    ia auto "o que você precisar"
    cat contrato.txt | ia analisar -s "Você é um advogado empresarial."
"""

from __future__ import annotations

import argparse
import sys

import anthropic

from .router import Roteador, SemRespostaValida, carregar_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ia",
        description="Usa o modelo Claude certo para cada função.",
    )
    parser.add_argument("funcao", help="'funcoes' para listar, 'auto' para escolher sozinho, ou o nome da função")
    parser.add_argument("texto", nargs="?", help="pedido (se omitido, lê da entrada padrão)")
    parser.add_argument("-s", "--system", help="instrução de sistema (papel, tom, regras)")
    parser.add_argument("-c", "--config", help="caminho para um rotas.toml personalizado")
    parser.add_argument("-v", "--verbose", action="store_true", help="mostra modelo, tokens e custo")
    args = parser.parse_args(argv)

    config = carregar_config(args.config)

    if args.funcao == "funcoes":
        for nome, f in config.funcoes.items():
            cadeia = f.modelo + (f" -> {f.escalar_para}" if f.escalar_para else "")
            print(f"{nome:<12} {cadeia:<18} effort={f.effort:<6} {f.descricao}")
        return 0

    texto = args.texto if args.texto is not None else sys.stdin.read()
    if not texto.strip():
        parser.error("informe o texto do pedido (argumento ou entrada padrão)")

    roteador = Roteador(config=config)
    try:
        if args.funcao == "auto":
            res = roteador.auto(texto, system=args.system)
        else:
            res = roteador.executar(args.funcao, texto, system=args.system)
    except KeyError as e:
        print(e.args[0], file=sys.stderr)
        return 2
    except SemRespostaValida as e:
        print(str(e), file=sys.stderr)
        return 3
    except anthropic.AuthenticationError:
        print("Chave da API inválida ou ausente. Defina ANTHROPIC_API_KEY.", file=sys.stderr)
        return 4
    except anthropic.APIConnectionError:
        print("Sem conexão com a API da Anthropic.", file=sys.stderr)
        return 5
    except anthropic.APIStatusError as e:
        print(f"Erro da API ({e.status_code}): {e.message}", file=sys.stderr)
        return 6

    print(res.texto)
    if res.truncado:
        print("\n[aviso] resposta cortada por limite de tamanho (max_tokens).", file=sys.stderr)
    if args.verbose:
        custo = f"US$ {res.custo_usd:.4f}" if res.custo_usd is not None else "n/d"
        caminho = " -> ".join(t.modelo for t in res.tentativas)
        print(
            f"\n[{res.funcao}] modelo={res.modelo} tentativas={caminho} "
            f"tokens={res.tokens_entrada}/{res.tokens_saida} custo={custo}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
