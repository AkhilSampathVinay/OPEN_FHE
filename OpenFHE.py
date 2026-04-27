"""
CS 645/745: Modern Cryptography — Spring 2026
FHE in AI and Large Language Models
OpenFHE (CKKS) — Linux/Codespace version
"""

import time
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Imports ───────────────────────────────────────────────────────────────────
try:
    import openfhe as fhe
    print("[INFO] OpenFHE loaded successfully")
except ImportError:
    print("[ERROR] OpenFHE not found. Run: pip3 install openfhe")
    exit(1)

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    HAS_TRANSFORMERS = True
    print("[INFO] Transformers + Torch loaded")
except ImportError:
    HAS_TRANSFORMERS = False
    print("[WARN] Transformers/Torch missing. Run: pip3 install transformers torch")

print()
DIV = "=" * 60

# ── Context Setup ─────────────────────────────────────────────────────────────
def make_context(batch_size=64):
    params = fhe.CCParamsCKKSRNS()
    params.SetMultiplicativeDepth(3)
    params.SetScalingModSize(50)
    params.SetBatchSize(batch_size)

    cc = fhe.GenCryptoContext(params)
    cc.Enable(fhe.PKESchemeFeature.PKE)
    cc.Enable(fhe.PKESchemeFeature.KEYSWITCH)
    cc.Enable(fhe.PKESchemeFeature.LEVELEDSHE)
    cc.Enable(fhe.PKESchemeFeature.ADVANCEDSHE)

    kp = cc.KeyGen()
    cc.EvalMultKeyGen(kp.secretKey)
    cc.EvalSumKeyGen(kp.secretKey)
    return cc, kp

# ── Encrypt / Decrypt ─────────────────────────────────────────────────────────
def encrypt(cc, kp, vec):
    pt = cc.MakeCKKSPackedPlaintext(vec)
    return cc.Encrypt(kp.publicKey, pt)

def decrypt(cc, kp, enc, length):
    pt = cc.Decrypt(enc, kp.secretKey)
    pt.SetLength(length)
    return list(pt.GetRealPackedValue())

# ── Plaintext conversion helper ───────────────────────────────────────────────
def to_pt(cc, plain_list):
    return cc.MakeCKKSPackedPlaintext(plain_list)

# ── Encrypted Operations ──────────────────────────────────────────────────────
def enc_add(cc, enc, plain_list):
    pt = cc.MakeCKKSPackedPlaintext(plain_list)
    return cc.EvalAdd(enc, pt)

def enc_mult(cc, enc, plain_list):
    pt = cc.MakeCKKSPackedPlaintext(plain_list)
    return cc.EvalMult(enc, pt)

def enc_dot(cc, kp, enc_vec, plain_weights):
    pt     = cc.MakeCKKSPackedPlaintext(plain_weights)
    scaled = cc.EvalMult(enc_vec, pt)
    return cc.EvalSum(scaled, len(plain_weights))

# ── Demo 1: Encrypted Dataset ─────────────────────────────────────────────────
def demo_dataset():
    print(DIV)
    print("  DEMO 1 — Encrypted Dataset: Addition & Multiplication")
    print(DIV)

    cc, kp = make_context(batch_size=8)

    dataset = [
        [0.72, 0.38, 0.91, 0.55],
        [0.14, 0.60, 0.33, 0.88],
        [0.50, 0.50, 0.50, 0.50],
    ]
    bias    = [0.10, 0.10, 0.10, 0.10]
    weights = [1.00, 2.00, 0.50, 1.50]

    print(f"\n  Dataset : {len(dataset)} samples, 4 features each")
    print(f"  Bias    : {bias}")
    print(f"  Weights : {weights}\n")

    for i, sample in enumerate(dataset):
        # Plaintext ground truth
        pt_add = [round(s + b, 6) for s, b in zip(sample, bias)]
        pt_mul = [round(s * w, 6) for s, w in zip(sample, weights)]

        # Encrypt
        enc = encrypt(cc, kp, sample)

        # Homomorphic operations (list → plaintext conversion handled inside)
        enc_added  = enc_add(cc, enc, bias)
        enc_scaled = enc_mult(cc, enc, weights)

        # Decrypt
        dec_add = [round(v, 6) for v in decrypt(cc, kp, enc_added,  4)]
        dec_mul = [round(v, 6) for v in decrypt(cc, kp, enc_scaled, 4)]

        # Correctness check
        err_add = max(abs(a - b) for a, b in zip(pt_add, dec_add))
        err_mul = max(abs(a - b) for a, b in zip(pt_mul, dec_mul))

        print(f"  Sample {i+1}: {sample}")
        print(f"    [ADD] Expected   : {pt_add}")
        print(f"    [ADD] FHE result : {dec_add}  err={err_add:.2e}  {'PASS' if err_add < 1e-4 else 'FAIL'}")
        print(f"    [MUL] Expected   : {pt_mul}")
        print(f"    [MUL] FHE result : {dec_mul}  err={err_mul:.2e}  {'PASS' if err_mul < 1e-4 else 'FAIL'}")
        print()

    print(DIV + "\n")

# ── Demo 2: Benchmark ─────────────────────────────────────────────────────────
def demo_benchmark(dim=64):
    print(DIV)
    print(f"  DEMO 2 — Performance Benchmark  (dim={dim})")
    print(DIV)

    np.random.seed(42)
    a = np.random.randn(dim).tolist()
    w = np.random.randn(dim).tolist()

    # Plaintext baseline
    t0 = time.perf_counter()
    for _ in range(10_000):
        _ = float(np.dot(a, w))
    pt_time = (time.perf_counter() - t0) / 10_000
    print(f"\n  Plaintext  (avg 10,000 runs)  : {pt_time*1000:.5f} ms")

    # Context + key setup
    t0 = time.perf_counter()
    cc, kp = make_context(batch_size=dim)
    print(f"  Key generation                : {(time.perf_counter()-t0)*1000:.2f} ms")

    # Encryption
    t0 = time.perf_counter()
    enc_a = encrypt(cc, kp, a)
    enc_ms = (time.perf_counter() - t0) * 1000
    print(f"  Encryption ({dim}-dim)          : {enc_ms:.2f} ms")

    # FHE dot product
    t0 = time.perf_counter()
    for _ in range(5):
        enc_r = enc_dot(cc, kp, enc_a, w)
    fhe_time = (time.perf_counter() - t0) / 5
    print(f"  FHE dot product (avg 5 runs)  : {fhe_time*1000:.2f} ms")

    # Decryption
    t0 = time.perf_counter()
    dec_val = decrypt(cc, kp, enc_r, dim)
    dec_ms = (time.perf_counter() - t0) * 1000
    print(f"  Decryption                    : {dec_ms:.2f} ms")

    # Results + correctness
    pt_dot  = float(np.dot(a, w))
    fhe_dot = dec_val[0]
    err     = abs(pt_dot - fhe_dot)

    print(f"\n  Plaintext result  : {pt_dot:.8f}")
    print(f"  FHE result        : {fhe_dot:.8f}")
    print(f"  Absolute error    : {err:.2e}  {'PASS' if err < 1e-3 else 'FAIL'}")
    print(f"  Overhead          : ~{fhe_time/pt_time:,.0f}x")
    print(f"  Total round-trip  : {enc_ms + fhe_time*1000 + dec_ms:.2f} ms")
    print(DIV + "\n")

# ── Demo 3: Privacy-Preserving Inference (+10%) ───────────────────────────────
def demo_private_inference():
    print(DIV)
    print("  DEMO 3 — Privacy-Preserving Inference  (+10%)")
    print(DIV)

    if not HAS_TRANSFORMERS:
        print("  [SKIPPED] Run: pip3 install transformers torch")
        print(DIV + "\n")
        return

    text = "The patient exhibits elevated cortisol and persistent insomnia."
    print(f"\n  Input: \"{text}\"")

    # Step 1: embed locally
    print("\n  [Step 1] Generating embedding (local — never sent to server)...")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    model      = AutoModel.from_pretrained(model_name)
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        out = model(**inputs)
    emb = out.last_hidden_state.mean(dim=1).squeeze().numpy().tolist()[:64]
    print(f"           Dim: 64  |  First 5: {[round(v, 4) for v in emb[:5]]}")

    # Step 2: encrypt
    print("\n  [Step 2] Encrypting embedding (secret key stays on device)...")
    cc, kp = make_context(batch_size=64)
    t0 = time.perf_counter()
    enc_emb = encrypt(cc, kp, emb)
    print(f"           Encryption time : {(time.perf_counter()-t0)*1000:.2f} ms")
    print("           Ciphertext sent to server.")

    # Step 3: server computes on ciphertext
    print("\n  [Step 3] Server classifies ENCRYPTED embedding...")
    np.random.seed(2026)
    w_pos = np.random.randn(64).tolist()
    w_neg = (-np.array(w_pos)).tolist()

    t0 = time.perf_counter()
    enc_pos = enc_dot(cc, kp, enc_emb, w_pos)
    enc_neg = enc_dot(cc, kp, enc_emb, w_neg)
    print(f"           Compute time    : {(time.perf_counter()-t0)*1000:.2f} ms")
    print("           Server never saw any plaintext.")

    # Step 4: user decrypts
    print("\n  [Step 4] User decrypts result...")
    s_pos = decrypt(cc, kp, enc_pos, 64)[0]
    s_neg = decrypt(cc, kp, enc_neg, 64)[0]

    pt_pos   = float(np.dot(emb, w_pos))
    pt_neg   = float(np.dot(emb, w_neg))
    label    = "POSITIVE" if s_pos > s_neg else "NEGATIVE"
    pt_label = "POSITIVE" if pt_pos > pt_neg else "NEGATIVE"

    print(f"\n  Positive | FHE: {s_pos:+.5f}  Plain: {pt_pos:+.5f}  err={abs(s_pos-pt_pos):.2e}")
    print(f"  Negative | FHE: {s_neg:+.5f}  Plain: {pt_neg:+.5f}  err={abs(s_neg-pt_neg):.2e}")
    print(f"  FHE label    : {label}")
    print(f"  Plain label  : {pt_label}")
    print(f"  Match        : {'YES — PASS' if label == pt_label else 'NO — FAIL'}")
    print(DIV + "\n")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n  CS 645/745 — FHE in AI Demo  |  OpenFHE CKKS (Linux)\n")
    demo_dataset()
    demo_benchmark()
    demo_private_inference()
    print("  All demos complete.")
