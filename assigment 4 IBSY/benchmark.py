import time
import tracemalloc
import random
from utils import alignment_identity, sum_of_pairs
from pairwise import (
    needleman_wunsch,
    smith_waterman,
    gotoh,
    hirschberg,
    blast_style,
    minimizer_align,
)
from msa import progressive_msa, iterative_refinement_msa, SimpleProfileHMM

DNA = "ACGT"
PROT = "ACDEFGHIKLMNPQRSTVWY"

def random_sequence(length: int, alphabet: str = DNA) -> str:
    return "".join(random.choice(alphabet) for _ in range(length))

def mutate_sequence(seq: str, snps: int = 5, indels: int = 2, alphabet: str = DNA) -> str:
    seq = list(seq)
    for _ in range(snps):
        if not seq:
            break
        i = random.randrange(len(seq))
        seq[i] = random.choice(alphabet)
    for _ in range(indels):
        if random.random() < 0.5 and len(seq) > 1:
            i = random.randrange(len(seq))
            seq.pop(i)
        else:
            i = random.randrange(len(seq) + 1)
            seq.insert(i, random.choice(alphabet))
    return "".join(seq)

def benchmark_call(func, *args, **kwargs):
    tracemalloc.start()
    t0 = time.perf_counter()
    result = func(*args, **kwargs)
    runtime = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, runtime, peak / 1024.0

def run_pairwise_benchmark():
    methods = {
        "Needleman-Wunsch": needleman_wunsch,
        "Smith-Waterman": smith_waterman,
        "Gotoh": gotoh,
        "Hirschberg": hirschberg,
        "BLAST-style": blast_style,
        "Minimizer": minimizer_align,
    }

    for L in [50, 100, 200]:
        s1 = random_sequence(L)
        s2 = mutate_sequence(s1, snps=max(1, L // 10), indels=max(1, L // 20))
        print(f"\n=== Length {L} ===")
        for name, func in methods.items():
            result, runtime, mem = benchmark_call(func, s1, s2)
            if isinstance(result, tuple) and len(result) == 3 and isinstance(result[1], str):
                score, a1, a2 = result
                ident = alignment_identity(a1, a2) if a1 and a2 else 0.0
                print(f"{name:20s} score={score:6} identity={ident:.3f} time={runtime:.4f}s mem={mem:.1f}KB")
            else:
                print(f"{name:20s} result={result} time={runtime:.4f}s mem={mem:.1f}KB")

def run_msa_demo():
    seqs = [mutate_sequence(random_sequence(40, PROT), snps=4, indels=1, alphabet=PROT) for _ in range(5)]

    msa1, t1, m1 = benchmark_call(progressive_msa, seqs)
    msa2, t2, m2 = benchmark_call(iterative_refinement_msa, seqs, 3)
    hmm = SimpleProfileHMM(msa1)
    (score, aligned), t3, m3 = benchmark_call(hmm.viterbi_align, seqs[0].replace("-", ""))

    print("\n=== MSA ===")
    print(f"Progressive MSA: score={sum_of_pairs(msa1)} time={t1:.4f}s mem={m1:.1f}KB")
    print(f"Iterative MSA:   score={sum_of_pairs(msa2)} time={t2:.4f}s mem={m2:.1f}KB")
    print(f"Profile HMM:     viterbi={score:.3f} time={t3:.4f}s mem={m3:.1f}KB")
    print("Example aligned sequence:", aligned)

if __name__ == "__main__":
    random.seed(42)
    run_pairwise_benchmark()
    run_msa_demo()