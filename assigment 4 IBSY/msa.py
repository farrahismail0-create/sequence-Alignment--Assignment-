from typing import List, Tuple, Dict
from math import log
from utils import consensus, sum_of_pairs, pad_msa
from pairwise import needleman_wunsch

LOG_ZERO = -10**9


def _profile_align(profile: List[str], seq: str) -> List[str]:
    cons = consensus(profile).replace("-", "")
    _, aln_cons, aln_seq = needleman_wunsch(cons, seq)

    ungapped_cols = [i for i in range(len(profile[0])) if any(s[i] != "-" for s in profile)]
    mapping = []
    raw_idx = 0

    for c in aln_cons:
        if c == "-":
            mapping.append(None)
        else:
            mapping.append(raw_idx)
            raw_idx += 1

    new_profile = []
    for p in profile:
        rebuilt = []
        for x in mapping:
            if x is None:
                rebuilt.append("-")
            else:
                rebuilt.append(p[ungapped_cols[x]])
        new_profile.append("".join(rebuilt))

    new_profile.append(aln_seq)
    return pad_msa(new_profile)


def progressive_msa(seqs: List[str]) -> List[str]:
    if len(seqs) <= 1:
        return seqs[:]

    profile = [seqs[0]]

    for seq in seqs[1:]:
        if len(profile) == 1:
            _, a, b = needleman_wunsch(profile[0], seq)
            profile = [a, b]
        else:
            profile = _profile_align(profile, seq)

    return pad_msa(profile)


def iterative_refinement_msa(seqs: List[str], iterations: int = 3) -> List[str]:
    msa = progressive_msa(seqs)
    best_score = sum_of_pairs(msa)

    for _ in range(iterations):
        cons = consensus(msa).replace("-", "")
        rebuilt = []

        for seq in seqs:
            _, _, aligned_seq = needleman_wunsch(cons, seq)
            rebuilt.append(aligned_seq)

        rebuilt = pad_msa(rebuilt)
        sc = sum_of_pairs(rebuilt)

        if sc > best_score:
            msa = rebuilt
            best_score = sc

    return msa


class SimpleProfileHMM:
    def __init__(self, msa: List[str], gap_threshold: float = 0.5, pseudo: float = 1.0):
        self.msa = pad_msa(msa)
        self.gap_threshold = gap_threshold
        self.pseudo = pseudo
        self.alphabet = sorted(set("".join(self.msa).replace("-", "")))
        self.match_cols = self._select_match_cols()
        self.emissions = self._estimate_emissions()

    def _select_match_cols(self) -> List[int]:
        cols = []
        n = len(self.msa)
        for j in range(len(self.msa[0])):
            gaps = sum(1 for s in self.msa if s[j] == "-")
            if gaps / n < self.gap_threshold:
                cols.append(j)
        return cols

    def _estimate_emissions(self) -> List[Dict[str, float]]:
        out = []
        for col in self.match_cols:
            counts = {a: self.pseudo for a in self.alphabet}
            for seq in self.msa:
                c = seq[col]
                if c != "-":
                    counts[c] += 1
            total = sum(counts.values())
            out.append({a: counts[a] / total for a in self.alphabet})
        return out

    def viterbi_align(self, seq: str) -> Tuple[float, str]:
        n = len(self.emissions)
        m = len(seq)
        dp = [[LOG_ZERO] * (m + 1) for _ in range(n + 1)]
        tb = [[None] * (m + 1) for _ in range(n + 1)]
        dp[0][0] = 0.0

        trans = log(0.8)
        gap = log(0.2)

        for i in range(1, n + 1):
            dp[i][0] = dp[i - 1][0] + gap
            tb[i][0] = "D"

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                emit = log(self.emissions[i - 1].get(seq[j - 1], 1e-9))
                match_val = dp[i - 1][j - 1] + trans + emit
                del_val = dp[i - 1][j] + gap
                ins_val = dp[i][j - 1] + gap

                best = max(match_val, del_val, ins_val)
                dp[i][j] = best
                tb[i][j] = "M" if best == match_val else "D" if best == del_val else "I"

        i, j = n, m
        aligned = []

        while i > 0 or j > 0:
            move = tb[i][j]
            if move == "M":
                aligned.append(seq[j - 1])
                i -= 1
                j -= 1
            elif move == "D":
                aligned.append("-")
                i -= 1
            elif move == "I":
                aligned.append(seq[j - 1].lower())
                j -= 1
            else:
                break

        return dp[n][m], "".join(reversed(aligned))