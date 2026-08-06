import time, torch, sys
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
try:
    import flash_kmeans as fk
    print("flash_kmeans imported:", [a for a in dir(fk) if not a.startswith("_")])
except Exception as e:
    print("IMPORT FAIL:", repr(e)); sys.exit(2)

dev = "cuda"
N, D, K = 2_000_000, 128, 200_000   # proxy for trec-covid K regime
x = torch.nn.functional.normalize(torch.randn(1, N, D, device=dev, dtype=torch.float16), dim=-1)
torch.cuda.synchronize(); t = time.time()
try:
    fn = getattr(fk, "batch_kmeans_Euclid", None)
    out = fn(x, n_clusters=K, max_iters=2, verbose=True) if fn else None
    torch.cuda.synchronize()
    print(f"OK K={K} N={N} in {time.time()-t:.1f}s; out types:", [type(o) for o in out] if isinstance(out,(tuple,list)) else type(out))
    print("peak GPU mem GB:", torch.cuda.max_memory_allocated()/1e9)
except Exception as e:
    print(f"RUN FAIL at K={K}:", repr(e))
