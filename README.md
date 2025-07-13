# TCC-2 – Integração YOLOv8 + OCR + LLM

Este projeto foi desenvolvido como parte do Trabalho de Conclusão de Curso 2, com o objetivo de integrar técnicas de visão computacional (YOLOv8), OCR (Tesseract) e validação semântica com modelo de linguagem (LLM) via Ollama.

## Estrutura de Arquivos

- `main.py` – pipeline principal do sistema
- `yolov8_detector.py` – carregamento e inferência com YOLOv8
- `ocr.py` – extração textual com Tesseract OCR
- `levenshtein.py` – cálculo de similaridade textual
- `llm.py` – consulta a modelo Mistral via API local
- `spreadsheet.py` – exportação de resultados para planilha Excel

## Execução

Certifique-se de que o Ollama está rodando localmente e o modelo Mistral está carregado.

## Requisitos
pip install ultralytics opencv-python pytesseract python-Levenshtein pandas requests

```bash
python main.py



