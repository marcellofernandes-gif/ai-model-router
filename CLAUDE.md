# Regras do projeto

- **Idioma:** sempre se comunique em português do Brasil (PT-BR) com qualquer pessoa que usar este repositório — respostas, explicações, mensagens de commit, descrições de PR e comentários no código.
- O mapa de funções -> modelo fica em `src/model_router/rotas.toml`. Ao mudar modelos ou preços, atualize também os testes em `tests/`.
- Rode `python -m pytest -q` antes de cada commit. Os testes não chamam a API (usam um cliente falso).
