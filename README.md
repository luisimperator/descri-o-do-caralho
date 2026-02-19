# Descrição Arbitragem

Automação de descrições estruturadas para vídeos do YouTube.

Transforma um link de vídeo em uma descrição profissional, utilizando extração de dados (yt-dlp), OCR de thumbnails (Tesseract) e validação de nomes via busca na internet.

## Requisitos

- Python 3.11+
- [yt-dlp](https://github.com/yt-dlp/yt-dlp)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (com modelos `por` e `eng`)

## Instalação

```bash
pip install -e .
```

Ou instale apenas as dependências:

```bash
pip install -r requirements.txt
```

### Tesseract (OCR)

```bash
# Ubuntu/Debian
sudo apt install tesseract-ocr tesseract-ocr-por tesseract-ocr-eng

# macOS
brew install tesseract tesseract-lang
```

## Uso

```bash
# Saída em texto (stdout)
python -m src https://www.youtube.com/watch?v=VIDEO_ID

# Salvar em arquivo
python -m src https://www.youtube.com/watch?v=VIDEO_ID -o descricao.txt

# Saída JSON completa
python -m src https://www.youtube.com/watch?v=VIDEO_ID --json

# Especificar diretório de trabalho
python -m src https://www.youtube.com/watch?v=VIDEO_ID -w /tmp/assets
```

Se instalado via `pip install -e .`:

```bash
descricao-arbitragem https://www.youtube.com/watch?v=VIDEO_ID
```

## Pipeline

1. **Extração de dados** — yt-dlp extrai metadados, thumbnail e transcrição (manual ou ASR)
2. **OCR** — Tesseract extrai texto da thumbnail (nomes, temas)
3. **Validação de nomes** — Candidatos extraídos de título, descrição, OCR e transcrição são validados via busca no Google (Protocolo Anti-Erro: mínimo 2 critérios)
4. **Geração de conteúdo** — Resumo (max 150 palavras), capítulos (existentes ou auto-segmentados a cada ~4min), keywords
5. **Template final** — Renderização no formato padronizado

## Template de Saída

```
[Título] | [Tópico Principal]

OCR: [texto curto da thumbnail]

No episódio de hoje, [Nomes] exploram [resumo].

Participantes
• Nome — Mini-bio

Tópicos Abordados:
00:00 Introdução
04:00 ...

Palavras-chave: termo1, termo2, ...

#Podcast #NomeDoPodcast #Tópico #Keyword
```

## Testes

```bash
pip install pytest
pytest
```

## Estrutura

```
src/
  __init__.py
  __main__.py       # Entry point para python -m src
  main.py           # CLI e orquestração do pipeline
  extractor.py      # Extração de dados via yt-dlp
  ocr.py            # OCR com Tesseract
  names.py          # Extração, validação e canonização de nomes
  content.py        # Geração de resumo, capítulos e keywords
  template.py       # Renderização do template final
tests/
  test_content.py
  test_ocr.py
  test_names.py
  test_template.py
```