from typing import List, Tuple

def score(a: str, b: str, match: int = 2, mismatch: int = -1) -> int:
    return match if a == b else mismatch

def alignment_identity(aln1: str, aln2: str) -> float:
    matches = 0
    total = 0
    for a, b in zip(aln1, aln2):
        if a != "-" and b != "-":
            total += 1
            if a == b:
                matches += 1
    return matches / total if total else 0.0

def pairwise_alignment_score(aln1: str, aln2: str, match=2, mismatch=-1, gap=-2) -> int:
    s = 0
    for a, b in zip(aln1, aln2):
        if a == "-" or b == "-":
            s += gap
        elif a == b:
            s += match
        else:
            s += mismatch
    return s

def sum_of_pairs(msa: List[str], match=2, mismatch=-1, gap=-2) -> int:
    total = 0
    n = len(msa)
    if n == 0:
        return 0
    L = len(msa[0])
    for col in range(L):
        for i in range(n):
            for j in range(i + 1, n):
                a, b = msa[i][col], msa[j][col]
                if a == "-" or b == "-":
                    total += gap
                elif a == b:
                    total += match
                else:
                    total += mismatch
    return total

def consensus(msa: List[str]) -> str:
    if not msa:
        return ""
    L = len(msa[0])
    cons = []
    for col in range(L):
        counts = {}
        for seq in msa:
            c = seq[col]
            counts[c] = counts.get(c, 0) + 1
        cons.append(max(counts, key=counts.get))
    return "".join(cons)

def pad_msa(msa: List[str]) -> List[str]:
    if not msa:
        return msa
    max_len = max(len(s) for s in msa)
    return [s + "-" * (max_len - len(s)) for s in msa]