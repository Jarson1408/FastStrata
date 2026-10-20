import urllib.request
import json
import time

def benchmark(prompt, max_tokens, reasoning_effort="none"):
    req = urllib.request.Request(
        "http://127.0.0.1:8080/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        data=json.dumps({
            "model": "qwen3.6-35b-a3b-ud-q4_k_m",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "reasoning_effort": reasoning_effort,
            "stream": True
        }).encode("utf-8")
    )
    t0 = time.time()
    t_first = None
    tokens = 0
    with urllib.request.urlopen(req, timeout=120) as r:
        for line in r:
            line = line.decode("utf-8").strip()
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            try:
                data = json.loads(data_str)
                delta = data["choices"][0]["delta"]
                content = delta.get("content") or delta.get("reasoning_content")
                if content:
                    if t_first is None:
                        t_first = time.time()
                    tokens += 1
            except Exception:
                pass
    t_end = time.time()
    ttft = (t_first - t0) if t_first else 0
    gen_time = (t_end - t_first) if t_first else 0
    tps = tokens / gen_time if gen_time > 0 else 0
    total_tps = tokens / (t_end - t0) if (t_end - t0) > 0 else 0
    return {
        "tokens": tokens,
        "ttft_s": ttft,
        "gen_time_s": gen_time,
        "tps": tps,
        "total_time_s": t_end - t0,
        "total_tps": total_tps
    }

print("=== Test 1: Standardowa odpowiedz (reasoning_effort=none, 256 tokens) ===")
res1 = benchmark("Opisz w kilku akapitach historie powstania internetu i jego wplyw na wspolczesny swiat.", 256, "none")
print(f"Wygenerowano tokenow: {res1['tokens']}")
print(f"Czas do pierwszego tokena (TTFT): {res1['ttft_s']:.3f} s")
print(f"Predkosc generowania (Decode TPS): {res1['tps']:.2f} tokenow/s")
print(f"Calkowity czas: {res1['total_time_s']:.2f} s")

print("\n=== Test 2: Dluzsza odpowiedz (reasoning_effort=none, 512 tokens) ===")
res2 = benchmark("Napisz szczegolowy esej o ewolucji algorytmow sztucznej inteligencji od logiki symbolicznej po sieci neuronowe i transformery.", 512, "none")
print(f"Wygenerowano tokenow: {res2['tokens']}")
print(f"Czas do pierwszego tokena (TTFT): {res2['ttft_s']:.3f} s")
print(f"Predkosc generowania (Decode TPS): {res2['tps']:.2f} tokenow/s")
print(f"Calkowity czas: {res2['total_time_s']:.2f} s")

print("\n=== Test 3: Tryb rozumowania (Reasoning / Thinking, 512 tokens) ===")
res3 = benchmark("Rozwiaz zadanie logiczne: Na farmie sa kury i owce. Razem maja 35 glow i 94 nogi. Ile jest kur, a ile owiec? Wyjasnij krok po kroku.", 512, "medium")
print(f"Wygenerowano tokenow: {res3['tokens']}")
print(f"Czas do pierwszego tokena (TTFT): {res3['ttft_s']:.3f} s")
print(f"Predkosc generowania (Decode TPS): {res3['tps']:.2f} tokenow/s")
print(f"Calkowity czas: {res3['total_time_s']:.2f} s")
