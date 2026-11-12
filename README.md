# model-router

Usa o modelo Claude certo para cada função: tarefas simples vão para o modelo rápido e barato, e tarefas que exigem raciocínio vão para o mais capaz. A qualidade é protegida por **escalonamento automático**: se o modelo mais barato recusar ou a resposta não passar na validação, o pedido sobe para o próximo modelo.

## Mapa padrão

| Função | Modelo | Effort | Se falhar, sobe para |
|---|---|---|---|
| `classificar` | Haiku 5.5 | low | Sonnet → Opus |
| `extrair` | Haiku 5.5 | medium | Sonnet → Opus |
| `resumir` | Sonnet 5.5 | medium | Opus |
| `traduzir` | Sonnet 5.5 | low | Opus |
| `redigir` | Sonnet 5.5 | medium | Opus |
| `atendimento` | Sonnet 5.5 | low | Opus |
| `codigo` | Opus 5.5 | high | — |
| `analisar` | Opus 5.5 | high | — |
| `planejar` | Opus 5.5 | high | — |

Preço por 1M de tokens (entrada/saída): Haiku 5.5 US$ 0,10/0,50 · Sonnet 5.5 US$ 2/10 · Opus 5.5 US$ 4/20.

Para mudar qualquer coisa, edite `src/model_router/rotas.toml` ou passe outro arquivo com `--config`.

### Como a qualidade é protegida

1. **Escalonamento:** quando há recusa, resposta vazia ou reprovação no `validar`, o pedido é refeito no próximo modelo da cadeia.
2. **Fallback no servidor:** no Opus 5.5 e no Sonnet 5.5, o parâmetro `fallbacks="default"` faz a própria API refazer o pedido em outro modelo se houver recusa por política de segurança.
3. **Modo `auto` conservador:** o Haiku escolhe a função. Se ele estiver em dúvida, o pedido vai para `analisar` (Opus).
4. **Raciocínio sempre ligado:** os modelos 5.5 usam pensamento adaptativo, e o `effort` controla a profundidade.

## Instalação

```bash
pip install -e ".[dev]"
```

```bash
export ANTHROPIC_API_KEY="sua-chave"
```

## Uso pela linha de comando

```bash
ia funcoes
```

```bash
ia resumir "texto longo aqui..." -v
```

```bash
cat contrato.txt | ia analisar -s "Você é um advogado empresarial. Aponte riscos." -v
```

```bash
ia auto "Escreva um e-mail convidando o time para a reunião de sexta"
```

Com `-v`, a saída também mostra o modelo usado, as tentativas, os tokens e o custo estimado.

## Uso no Python

```python
from model_router import Roteador

r = Roteador()

res = r.executar("classificar", "Cliente quer cancelar. Prioridade: baixa, média ou alta?")
print(res.texto, res.modelo, res.custo_usd)

# Validação: se não for JSON válido, sobe para o próximo modelo
import json
res = r.executar("extrair", "Extraia {empresa, cnpj} em JSON: ...", validar=lambda t: bool(json.loads(t)))

# Automático
res = r.auto("Revise esta query SQL: SELECT ...")
print(res.funcao, res.modelo, res.tentativas)
```

Veja mais em `examples/exemplo.py`.

## Testes

```bash
python -m pytest -q
```

Os testes usam um cliente falso e não gastam créditos da API.
