import os
import json
import time
from datasets import load_from_disk
from google import genai
from google.genai import types

from test_adapter import TestAdapter
from util import build_prompt

# =====================================================================
# CONFIGURAÇÃO DA API 
# =====================================================================
api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    api_key = "CHAVE_API" 

client = genai.Client(api_key=api_key)

def _from_hf_data(data):
    return {
        'focal_fn_signature': data['signature'],
        'docstring': data['docstring'],
        'focal_ctx': data['extended_context'],
        'rag_data': data.get('rag_data', []),
    }

# =====================================================================
# LOOP AGÊNTICO 
# =====================================================================
def gerar_e_testar_codigo(item_dataset, crate_base):
    prompt_usuario = build_prompt(_from_hf_data(item_dataset), True)
    
    historico_feedbacks = []
    melhor_codigo = ""
    melhor_nota = -1
    codigo_anterior = ""
    
    item_dict = dict(item_dataset)
    
    for i in range(10):
        print(f"\n  [Iteração {i+1}/10]")


        if not historico_feedbacks:
            prompt_gerador = prompt_usuario
        else:
            prompt_gerador = prompt_gerador = f"{prompt_usuario}\n\nCÓDIGO ANTERIOR:\n{codigo_anterior}\n\nERROS/FEEDBACK:\n{historico_feedbacks[-1]}\n\nIMPORTANTE: Retorne APENAS o código Rust. Não escreva nenhuma explicação ou texto conversacional."

        try:
            resposta = client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=prompt_gerador,
                config=types.GenerateContentConfig(
                    system_instruction="Você é um Desenvolvedor Rust. Gere apenas o código Rust cru para resolver o problema. Não use markdown.",
                    temperature=0.2
                )
            )
            codigo_atual = resposta.text.replace("```rust", "").replace("```", "").replace("[CODE]", "").replace("[/CODE]", "").strip()
            # --- Corta qualquer conversa antes do código ---
            if "pub fn" in codigo_atual:
                codigo_atual = codigo_atual[codigo_atual.find("pub fn"):]
            elif "fn " in codigo_atual:
                codigo_atual = codigo_atual[codigo_atual.find("fn "):]
            # -------------------------------------------------------------
        except Exception as e:
            print(f"  Erro na API (Gerador): {e}")
            time.sleep(5)
            continue
        
        if not codigo_atual.startswith('fn') and not codigo_atual.startswith('pub fn'):
            codigo_atual = item_dict['signature'] + ' ' + codigo_atual
            
        item_dict['focal_fn_full'] = codigo_atual 
        
        compilou = False
        try:
            with TestAdapter(crate_base, item_dict, replace_test=False) as adapter:
                compilou = adapter.compile()
        except Exception as e:
            print(f"  Erro ao compilar localmente: {e}")
        
        print(f"  -> Compilou no Cargo? {compilou}")
        
        prompt_juiz = f"PROMPT: {prompt_usuario}\nCÓDIGO: {codigo_atual}\nCOMPILOU? {compilou}"
        try:
            resposta_juiz = client.models.generate_content(
                model="gemini-3.1-flash-lite",
                contents=prompt_juiz,
                config=types.GenerateContentConfig(
                    system_instruction='Você é um Revisor Rust. Avalie o código. Retorne APENAS um JSON: {"nota": numero, "feedback_critico": "texto"}',
                    response_mime_type="application/json",
                    temperature=0.1
                )
            )
            dados_juiz = json.loads(resposta_juiz.text)
            nota = dados_juiz.get("nota", 0)
            feedback = dados_juiz.get("feedback_critico", "")
        except Exception as e:
            print(f"  Erro na API (Juiz): {e}")
            nota = 0
            feedback = "Erro ao avaliar."
            
        print(f"  -> Nota do Juiz: {nota}/10 | Feedback: {feedback}")
        
        historico_feedbacks.append(feedback)
        
        if nota > melhor_nota:
            melhor_nota = nota
            melhor_codigo = codigo_atual
            codigo_anterior = codigo_atual

            
        if compilou and nota >= 9:
            print("  ✓ Código compilou e passou na revisão!")
            break
            
        time.sleep(4) 
        
    return melhor_codigo

# =====================================================================
# EXECUÇÃO PRINCIPAL
# =====================================================================
def main():
    DATASET_PATH = "./dataset/rusteval" 
    CRATE_BASE = "./crates" 
    OUTPUT_PATH = "results/rusteval_xc/models-gemini-3.1-flash-lite_n10.json"
    
    print(f"Carregando dataset do HuggingFace em {DATASET_PATH}...")
    hf_dataset = load_from_disk(DATASET_PATH)
    
    hf_dataset = hf_dataset.select(range(10))
    
    resultados_finais = {
        "benchmark": "rusteval_xc",
        "n": 10,
        "k": [1, 10],
        "fn_codes": []
    }
    
    for idx, item in enumerate(hf_dataset):
        print(f"\n======================================")
        print(f"AVALIANDO PROBLEMA {idx+1}/{len(hf_dataset)}")
        print(f"======================================")
        
        codigo_vencedor = gerar_e_testar_codigo(item, CRATE_BASE)
        
        resultados_finais["fn_codes"].append([codigo_vencedor] * 10)
        
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(resultados_finais, f, indent=4)
        
    print(f"\nAvaliação concluída! Resultados salvos em {OUTPUT_PATH}")

if __name__ == "__main__":
    main()