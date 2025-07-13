import requests

def query_llm(yolo_class_name, ocr_results):
    flat_words = list({word for sublist in ocr_results for word in sublist})
    prompt = f"""
Você é um robô especialista em reconhecimento de objetos e em identificação de produtos através de uma lista OCR lida nos rótulos de objetos que ficam em prateleiras de supermercado.

O sistema de visão computacional (YOLO) detectou o seguinte produto:
- Produto: {yolo_class_name}

Palavras extraídas da embalagem via OCR (com possíveis erros):
{flat_words}

Com base nas palavras OCR, avalie se o produto identificado pelo YOLO está correto.

Responda:
1. O produto identificado pelo YOLO está correto? (responda apenas com "sim" ou "não")
2. Se a resposta for "não", informe o nome do produto correto com base nas palavras OCR (responda de forma curta e direta).
"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "mistral", "prompt": prompt, "stream": False, "temperature": 0.5}
        )
        if response.status_code == 200:
            reply = response.json().get("response", "").strip().lower()
            if "sim" in reply:
                return reply, 0.35
            elif "não" in reply:
                return reply, -0.2
            else:
                return reply, 0.0
        else:
            return f"LLM error: status {response.status_code}", 0.0
    except Exception as e:
        return f"LLM connection error: {str(e)}", 0.0
