"""Chapter 4: how far does exact (brute-force) search go?"""

import time

import numpy as np

from halden.retrieval.brute_force import top_k_cosine

rng = np.random.default_rng(7)
DIMS = 384
for n in (10_000, 100_000, 1_000_000):
    docs = rng.standard_normal((n, DIMS), dtype=np.float32)
    docs /= np.linalg.norm(docs, axis=1, keepdims=True)
    query = docs[0]
    runs = 20
    start = time.perf_counter()
    for _ in range(runs):
        top_k_cosine(query, docs, k=10)
    ms = (time.perf_counter() - start) / runs * 1000
    mb = docs.nbytes / 1e6
    print(f"n={n:>9,}  memory={mb:>7.0f} MB  query={ms:6.1f} ms")
