"""Exemplo de uso. Requer ANTHROPIC_API_KEY definida."""

import json

from model_router import Roteador

r = Roteador()

# 1) Função explícita -> modelo definido em rotas.toml
res = r.executar("classificar", "Cliente: 'o boleto veio errado de novo, quero cancelar'. "
                 "Responda só com: baixa, média ou alta prioridade.")
print(res.texto, "|", res.modelo, "| US$", res.custo_usd)


# 2) Com validação: se a resposta não for JSON válido, sobe para o próximo modelo
def eh_json(texto: str) -> bool:
    json.loads(texto)
    return True


res = r.executar(
    "extrair",
    "Extraia em JSON {empresa, cnpj, valor}: 'NF emitida por Acme Ltda, "
    "CNPJ 12.345.678/0001-90, total R$ 1.250,00'. Responda só com o JSON.",
    validar=eh_json,
)
print(res.texto, "| tentativas:", [t.modelo for t in res.tentativas])

# 3) Automático: o roteador decide a função (e portanto o modelo)
res = r.auto("Escreva um e-mail curto convidando o time para a reunião de resultados de sexta.")
print(f"[{res.funcao} / {res.modelo}]\n{res.texto}")
