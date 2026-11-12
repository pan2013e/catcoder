import hashlib
import json
import re

from .cache import Cache
from .limites import (
    FX_BRL,
    MAX_CHARS_DOC_SIG,
    MAX_CHARS_TIPO,
    MAX_CHARS_TRECHO,
    MAX_CHARS_VERSAO,
    MAX_CHARS_VERSAO_JUIZA,
    MAX_GRAU,
    MAX_OUT_GERADORA,
    MAX_OUT_JUIZA,
    MAX_OUT_N,
    MAX_TENTATIVAS,
    MAX_TRECHOS,
    MAX_VOLTAS,
    MODELO_N,
    TETO_BRL,
    custo_usd,
)


def cortar(texto, limite):
    texto = "" if texto is None else str(texto)
    return texto[:limite]


def trechos_rag(rag):
    if rag is None:
        return ""
    if isinstance(rag, list):
        partes = [str(p).strip() for p in rag if str(p).strip()]
    else:
        texto = str(rag).strip()
        partes = [p.strip() for p in texto.split("\n\n") if p.strip()]
        if len(partes) <= 1 and len(texto) > MAX_CHARS_TRECHO:
            partes = [texto[i:i + MAX_CHARS_TRECHO] for i in range(0, len(texto), MAX_CHARS_TRECHO)]
    return "\n\n".join(p[:MAX_CHARS_TRECHO] for p in partes[:MAX_TRECHOS])


def doc_e_assinatura(doc, assinatura):
    doc = cortar(doc, MAX_CHARS_DOC_SIG)
    assinatura = cortar(assinatura, max(0, MAX_CHARS_DOC_SIG - len(doc)))
    return doc, assinatura


def limpar_codigo(texto, lingua, assinatura):
    texto = texto or ""
    if "[CODE]" in texto:
        texto = texto.split("[CODE]", 1)[1]
    if "[/CODE]" in texto:
        texto = texto.split("[/CODE]", 1)[0]
    linhas = [ln for ln in texto.splitlines() if not ln.strip().startswith("```")]
    texto = "\n".join(linhas).strip()
    if lingua == "rust":
        for marca in ("pub fn", "fn "):
            pos = texto.find(marca)
            if pos != -1:
                texto = texto[pos:]
                break
        if texto and not texto.startswith("fn") and not texto.startswith("pub fn") and assinatura:
            texto = assinatura.strip() + " " + texto
    elif lingua == "java" and texto and assinatura:
        if not texto.startswith(("public", "private", "protected", "static", "@")):
            if not texto.startswith("{"):
                texto = "{\n" + texto
                if not texto.rstrip().endswith("}"):
                    texto = texto + "\n}"
            texto = assinatura.strip() + " " + texto
    return cortar(texto, MAX_CHARS_VERSAO)


def _sha(texto):
    return hashlib.sha1((texto or "").encode("utf-8")).hexdigest()


def _inteiro(texto):
    achado = re.search(r"-?\d+", texto or "")
    if not achado:
        return 2
    return max(0, min(MAX_GRAU, int(achado.group(0))))


def _json_juiza(texto):
    bruto = (texto or "").strip()
    if bruto.startswith("```"):
        bruto = bruto.split("\n", 1)[-1]
        if "```" in bruto:
            bruto = bruto.rsplit("```", 1)[0]
    dados = json.loads(bruto)
    acao = str(dados.get("acao", "")).lower()
    if acao not in ("continua", "reverte", "parar"):
        raise ValueError(acao)
    return {
        "acao": acao,
        "versao": int(dados.get("versao", 0)),
        "feedback": str(dados.get("feedback", ""))[:1500],
    }


class OpenAICliente:
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI()

    def completar(self, *, modelo, system, user, max_output_tokens):
        mensagens = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        kwargs = {"model": modelo, "messages": mensagens}
        if str(modelo).startswith("gpt-6"):
            kwargs["max_completion_tokens"] = max_output_tokens
            kwargs["reasoning_effort"] = "none"
            kwargs["temperature"] = 0.2
        elif str(modelo).startswith("gpt-5"):
            kwargs["max_completion_tokens"] = max_output_tokens
            kwargs["reasoning_effort"] = "minimal"
        else:
            kwargs["max_tokens"] = max_output_tokens
            kwargs["temperature"] = 0.2
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            msg = str(exc).lower()
            if "max_tokens" in msg or "max_completion" in msg or "temperature" in msg or "reasoning_effort" in msg:
                kwargs.pop("temperature", None)
                kwargs.pop("max_tokens", None)
                if "reasoning_effort" in msg:
                    kwargs.pop("reasoning_effort", None)
                kwargs["max_completion_tokens"] = max_output_tokens
                resp = self.client.chat.completions.create(**kwargs)
            else:
                raise
        texto = resp.choices[0].message.content or ""
        uso = resp.usage
        entrada = int(getattr(uso, "prompt_tokens", 0) or 0)
        saida = int(getattr(uso, "completion_tokens", 0) or 0)
        usd = custo_usd(modelo, entrada, saida)
        return {
            "texto": texto,
            "input_tokens": entrada,
            "output_tokens": saida,
            "usd": usd,
            "brl": usd * FX_BRL,
            "modelo": modelo,
        }


def chamar(cliente, cache: Cache, chave, *, modelo, system, user, max_output_tokens, teto_brl=TETO_BRL):
    achado = cache.ler(chave)
    if achado is not None:
        return achado, False
    if cache.medido_brl() >= teto_brl:
        raise RuntimeError("teto de R$ 20 atingido")
    ultimo = None
    for _ in range(MAX_TENTATIVAS):
        try:
            registro = cliente.completar(
                modelo=modelo,
                system=system,
                user=user,
                max_output_tokens=max_output_tokens,
            )
            cache.gravar(chave, registro)
            return registro, True
        except RuntimeError:
            raise
        except Exception as exc:
            ultimo = exc
    raise RuntimeError(f"falha apos {MAX_TENTATIVAS} tentativas: {ultimo}")


def escolher_n(cliente, cache, lingua, indice, doc, assinatura):
    doc, assinatura = doc_e_assinatura(doc, assinatura)
    user = f"{doc}\n{assinatura}".strip()
    chave = {
        "lingua": lingua, "indice": indice, "papel": "n",
        "modelo": MODELO_N, "volta": 0, "prompt": _sha(user),
    }
    registro, novo = chamar(
        cliente, cache, chave,
        modelo=MODELO_N,
        system="Responda somente um inteiro de 0 a 4. Ele é o grau do grafo de tipos.",
        user=user,
        max_output_tokens=MAX_OUT_N,
    )
    pedido = _inteiro(registro["texto"])
    usado = max(0, min(MAX_GRAU, pedido))
    return pedido, usado, registro, novo


def _prompt_juiza(prompt, versoes):
    blocos = [prompt, "", "VERSÕES:"]
    for item in versoes[-10:]:
        blocos.append(f"id={item['id']}")
        blocos.append(cortar(item["codigo"], MAX_CHARS_VERSAO_JUIZA))
    blocos.append(
        'Responda só JSON: {"acao":"continua"|"reverte"|"parar","versao":id,"feedback":"texto"}'
    )
    return "\n".join(blocos)


def _prompt_geradora(prompt, versao):
    if versao is None:
        return prompt
    return (
        f"{prompt}\n\nVERSÃO A REVISAR (id={versao['id']}):\n{versao['codigo']}"
        f"\n\nFEEDBACK:\n{versao.get('feedback', '')}\n\n"
        "Devolva só o código revisado."
    )


def _somar(acumulado, registro):
    acumulado["in"] += int(registro.get("input_tokens") or 0)
    acumulado["out"] += int(registro.get("output_tokens") or 0)
    acumulado["brl"] += float(registro.get("brl") or 0)


def trajetoria(cliente, cache, *, lingua, indice, rodada, geradora, juiza, doc, assinatura, montar_prompt, testar):
    n_pedido, n_usado, registro_n, novo_n = escolher_n(
        cliente, cache, lingua, indice, doc, assinatura
    )
    prompt = montar_prompt(n_usado)
    versoes = []
    acoes = []
    atual = None
    conta = {"in": 0, "out": 0, "brl": 0.0}
    if novo_n:
        _somar(conta, registro_n)
    acao_final = "parar"
    entrega_id = 0

    for volta in range(1, MAX_VOLTAS + 1):
        user_g = _prompt_geradora(prompt, atual)
        registro_g, novo_g = chamar(
            cliente, cache,
            {
                "lingua": lingua, "indice": indice, "rodada": rodada,
                "papel": "geradora", "volta": volta, "modelo": geradora,
                "prompt": _sha(user_g),
            },
            modelo=geradora,
            system=(
                "Você completa a função pedida. Devolva só o código, sem markdown."
            ),
            user=user_g,
            max_output_tokens=MAX_OUT_GERADORA,
        )
        if novo_g:
            _somar(conta, registro_g)
        codigo = limpar_codigo(registro_g["texto"], lingua, assinatura)
        versao = {"id": len(versoes), "codigo": codigo, "feedback": ""}
        versoes.append(versao)
        atual = versao

        user_j = _prompt_juiza(prompt, versoes)
        registro_j, novo_j = chamar(
            cliente, cache,
            {
                "lingua": lingua, "indice": indice, "rodada": rodada,
                "papel": "juiza", "volta": volta, "modelo": juiza,
                "prompt": _sha(user_j),
            },
            modelo=juiza,
            system=(
                "Você é a juíza. Escolha qual versão reavaliar. "
                "Responda somente o JSON pedido."
            ),
            user=user_j,
            max_output_tokens=MAX_OUT_JUIZA,
        )
        if novo_j:
            _somar(conta, registro_j)
        try:
            decisao = _json_juiza(registro_j["texto"])
        except (ValueError, json.JSONDecodeError, TypeError):
            decisao = {"acao": "parar", "versao": versao["id"], "feedback": ""}
        acoes.append(decisao["acao"])
        acao_final = decisao["acao"]
        alvo = decisao["versao"]
        if not (0 <= alvo < len(versoes)):
            alvo = versao["id"]
        versoes[alvo]["feedback"] = decisao["feedback"]
        entrega_id = alvo
        if decisao["acao"] == "parar":
            break
        if decisao["acao"] == "reverte":
            atual = versoes[alvo]
        else:
            atual = versao

    entregue = versoes[entrega_id]["codigo"]
    compilou, passou = testar(entregue)
    return {
        "n_pedido": n_pedido,
        "n_usado": n_usado,
        "voltas": len(acoes),
        "acoes": acoes,
        "acao_final": acao_final,
        "codigo": entregue,
        "compilou": bool(compilou),
        "passou": bool(passou),
        "tokens_in": conta["in"],
        "tokens_out": conta["out"],
        "brl": conta["brl"],
        "versoes": [{"id": v["id"], "codigo": v["codigo"]} for v in versoes],
    }
