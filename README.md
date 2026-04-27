# Fully Homomorphic Encryption (FHE) in AI & Large Language Models

> CS 645/745: Modern Cryptography — Spring 2026  
> University of Alabama at Birmingham  
> Instructor: Dr. Yuliang Zheng

---

## Overview

This project demonstrates how **Fully Homomorphic Encryption (FHE)** can be applied to secure AI computations, specifically in the context of Large Language Models (LLMs). Using **OpenFHE's CKKS scheme**, it shows how sensitive data can be encrypted, computed on, and decrypted — without ever exposing the plaintext to the processing party.

---

## What is FHE?

Fully Homomorphic Encryption allows arbitrary mathematical operations to be performed **directly on encrypted data** (ciphertexts). The decrypted result matches exactly what you would get by running the same operations on the original plaintext.

| Property | Traditional Encryption | FHE (CKKS) |
|---|---|---|
| Compute on ciphertext | ❌ Not possible | ✅ Full support |
| Plaintext exposed to server | ✅ Yes | ❌ Never |
| ML inference support | ❌ No | ✅ Yes |
| Overhead vs plaintext | ~1x | ~15,000x |

---

## Project Structure

```
├── FHE_demo.py        # Main demo script (OpenFHE CKKS)
├── README.md          # This file
```

---

## Demos

### Demo 1 — Encrypted Dataset: Addition & Multiplication
Encrypts three 4-feature samples and applies bias addition and weight multiplication **homomorphically**. Decrypted results are verified against NumPy ground truth.

**Output:**
```
Sample 1: [0.72, 0.38, 0.91, 0.55]
  [ADD] Expected   : [0.82, 0.48, 1.01, 0.65]
  [ADD] FHE result : [0.82, 0.48, 1.01, 0.65]  err=0.00e+00  PASS
  [MUL] Expected   : [0.72, 0.76, 0.455, 0.825]
  [MUL] FHE result : [0.72, 0.76, 0.455, 0.825]  err=0.00e+00  PASS
```

---

### Demo 2 — Performance Benchmark (dim=64)
Measures and compares plaintext vs. FHE operation timing on a 64-dimensional vector.

**Actual results (GitHub Codespace, Linux):**
```
Plaintext  (avg 10,000 runs)  : 0.00812 ms
Key generation                : 247.27 ms
Encryption (64-dim)           : 15.11 ms
FHE dot product (avg 5 runs)  : 124.68 ms
Decryption                    : 15.94 ms

Plaintext result  : 1.18198354
FHE result        : 1.18198354
Absolute error    : 1.34e-12   PASS
Overhead          : ~15,351x
Total round-trip  : 155.73 ms
```

---

### Demo 3 — Privacy-Preserving Inference (+10% Challenge)
Full end-to-end pipeline where:
1. User encodes text → embedding **locally** (never leaves device)
2. User **encrypts** embedding with CKKS secret key
3. Server classifies the **encrypted** embedding (never sees plaintext)
4. Server returns **encrypted** scores
5. User **decrypts** and reads the label

**Actual results:**
```
Input: "The patient exhibits elevated cortisol and persistent insomnia."

Positive | FHE: -1.19910  Plain: -1.19910  err=2.29e-12
Negative | FHE: +1.19910  Plain: +1.19910  err=2.63e-12
FHE label    : NEGATIVE
Plain label  : NEGATIVE
Match        : YES — PASS
```

---

## Setup & Installation

### Requirements
- Linux or GitHub Codespace (Ubuntu 22.04 recommended)
- Python 3.8+

> **Note:** OpenFHE does not provide Mac wheels. Use GitHub Codespaces or any Linux environment.

### Install dependencies
```bash
pip3 install openfhe numpy transformers torch
```

### Verify installation
```bash
python3 -c "import openfhe, transformers, torch, numpy; print('All imports OK')"
```

### Run
```bash
python3 FHE_demo.py
```

---

## Key Code Snippets

### CKKS Context Setup
```python
import openfhe as fhe

params = fhe.CCParamsCKKSRNS()
params.SetMultiplicativeDepth(3)
params.SetScalingModSize(50)
params.SetBatchSize(64)

cc = fhe.GenCryptoContext(params)
cc.Enable(fhe.PKESchemeFeature.PKE)
cc.Enable(fhe.PKESchemeFeature.KEYSWITCH)
cc.Enable(fhe.PKESchemeFeature.LEVELEDSHE)
cc.Enable(fhe.PKESchemeFeature.ADVANCEDSHE)  # required for EvalSum

kp = cc.KeyGen()
cc.EvalMultKeyGen(kp.secretKey)
cc.EvalSumKeyGen(kp.secretKey)
```

### Encrypt & Decrypt
```python
def encrypt(cc, kp, vec):
    pt = cc.MakeCKKSPackedPlaintext(vec)
    return cc.Encrypt(kp.publicKey, pt)

def decrypt(cc, kp, enc, length):
    pt = cc.Decrypt(enc, kp.secretKey)
    pt.SetLength(length)
    return list(pt.GetRealPackedValue())
```

### Homomorphic Operations
```python
# Lists must be converted to Plaintext before EvalAdd/EvalMult
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
```

---

## Results Summary

| Demo | Result | Error | Status |
|---|---|---|---|
| Encrypted Addition | Matches plaintext | 0.00e+00 | ✅ PASS |
| Encrypted Multiplication | Matches plaintext | 0.00e+00 | ✅ PASS |
| FHE Dot Product | 1.18198354 | 1.34e-12 | ✅ PASS |
| Private Inference Label | NEGATIVE | 2.29e-12 | ✅ PASS |

---

## Limitations & Trade-offs

| Approach | Privacy | Speed | Best For |
|---|---|---|---|
| Full FHE | Maximum | ~15,000x slower | Linear layers only |
| Secure Enclave (TEE) | High | Near-native | Full model |
| Hybrid FHE + TEE | High | Moderate | Most practical today |
| Differential Privacy | Statistical | ~1x | Training only |

**Key limitations of FHE for full LLM inference:**
- Nonlinear functions (softmax, GeLU) require high-degree polynomial approximations
- Memory usage grows significantly with model size
- Interactive latency is not yet feasible for GPT-scale models

---

## References

- Gentry, C. (2009). *A fully homomorphic encryption scheme.* Stanford PhD Thesis.
- Cheon et al. (2017). *Homomorphic encryption for arithmetic of approximate numbers.* ASIACRYPT.
- Al Badawi et al. (2022). *OpenFHE: Open-source fully homomorphic encryption library.* ACM CCS.
- [OpenFHE Documentation](https://openfhe.org)
- [DARPA DPRIVE Program](https://www.darpa.mil/program/dprive)

---

## License

This project was created for academic purposes as part of CS 645/745 at UAB. Not intended for production use.
