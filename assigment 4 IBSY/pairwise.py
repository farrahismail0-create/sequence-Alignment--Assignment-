from typing import List, Tuple, Dict
from utils import score, pairwise_alignment_score

NEG_INF = -10**9

def needleman_wunsch(seq1: str, seq2: str, match=2, mismatch=-1, gap=-2) -> Tuple[int, str, str]:
    n, m = len(seq1), len(seq2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    tb = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i * gap
        tb[i][0] = "U"
    for j in range(1, m + 1):
        dp[0][j] = j * gap
        tb[0][j] = "L"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = dp[i - 1][j - 1] + score(seq1[i - 1], seq2[j - 1], match, mismatch)
            up = dp[i - 1][j] + gap
            left = dp[i][j - 1] + gap
            best = max(diag, up, left)
            dp[i][j] = best
            tb[i][j] = "D" if best == diag else "U" if best == up else "L"

    i, j = n, m
    a1, a2 = [], []
    while i > 0 or j > 0:
        move = tb[i][j]
        if move == "D":
            a1.append(seq1[i - 1])
            a2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif move == "U":
            a1.append(seq1[i - 1])
            a2.append("-")
            i -= 1
        else:
            a1.append("-")
            a2.append(seq2[j - 1])
            j -= 1

    return dp[n][m], "".join(reversed(a1)), "".join(reversed(a2))

def smith_waterman(seq1: str, seq2: str, match=2, mismatch=-1, gap=-2) -> Tuple[int, str, str]:
    n, m = len(seq1), len(seq2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    tb = [[None] * (m + 1) for _ in range(n + 1)]

    best_score = 0
    best_pos = (0, 0)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = dp[i - 1][j - 1] + score(seq1[i - 1], seq2[j - 1], match, mismatch)
            up = dp[i - 1][j] + gap
            left = dp[i][j - 1] + gap
            best = max(0, diag, up, left)
            dp[i][j] = best

            if best == 0:
                tb[i][j] = None
            elif best == diag:
                tb[i][j] = "D"
            elif best == up:
                tb[i][j] = "U"
            else:
                tb[i][j] = "L"

            if best > best_score:
                best_score = best
                best_pos = (i, j)

    i, j = best_pos
    a1, a2 = [], []
    while i > 0 and j > 0 and dp[i][j] > 0:
        move = tb[i][j]
        if move == "D":
            a1.append(seq1[i - 1])
            a2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif move == "U":
            a1.append(seq1[i - 1])
            a2.append("-")
            i -= 1
        elif move == "L":
            a1.append("-")
            a2.append(seq2[j - 1])
            j -= 1
        else:
            break

    return best_score, "".join(reversed(a1)), "".join(reversed(a2))

def gotoh(seq1: str, seq2: str, match=2, mismatch=-1, gap_open=-3, gap_extend=-1) -> Tuple[int, str, str]:
    n, m = len(seq1), len(seq2)
    M = [[NEG_INF] * (m + 1) for _ in range(n + 1)]
    X = [[NEG_INF] * (m + 1) for _ in range(n + 1)]
    Y = [[NEG_INF] * (m + 1) for _ in range(n + 1)]

    tbM = [[None] * (m + 1) for _ in range(n + 1)]
    tbX = [[None] * (m + 1) for _ in range(n + 1)]
    tbY = [[None] * (m + 1) for _ in range(n + 1)]

    M[0][0] = 0
    for i in range(1, n + 1):
        X[i][0] = gap_open + (i - 1) * gap_extend
        tbX[i][0] = "X"
    for j in range(1, m + 1):
        Y[0][j] = gap_open + (j - 1) * gap_extend
        tbY[0][j] = "Y"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = score(seq1[i - 1], seq2[j - 1], match, mismatch)

            prevs = [("M", M[i - 1][j - 1]), ("X", X[i - 1][j - 1]), ("Y", Y[i - 1][j - 1])]
            prev_state, prev_val = max(prevs, key=lambda x: x[1])
            M[i][j] = prev_val + s
            tbM[i][j] = prev_state

            open_x = M[i - 1][j] + gap_open
            ext_x = X[i - 1][j] + gap_extend
            if open_x >= ext_x:
                X[i][j] = open_x
                tbX[i][j] = "M"
            else:
                X[i][j] = ext_x
                tbX[i][j] = "X"

            open_y = M[i][j - 1] + gap_open
            ext_y = Y[i][j - 1] + gap_extend
            if open_y >= ext_y:
                Y[i][j] = open_y
                tbY[i][j] = "M"
            else:
                Y[i][j] = ext_y
                tbY[i][j] = "Y"

    states = {"M": M[n][m], "X": X[n][m], "Y": Y[n][m]}
    state = max(states, key=states.get)
    best = states[state]

    i, j = n, m
    a1, a2 = [], []
    while i > 0 or j > 0:
        if state == "M":
            prev = tbM[i][j]
            a1.append(seq1[i - 1])
            a2.append(seq2[j - 1])
            i -= 1
            j -= 1
            state = prev
        elif state == "X":
            prev = tbX[i][j]
            a1.append(seq1[i - 1])
            a2.append("-")
            i -= 1
            state = prev
        elif state == "Y":
            prev = tbY[i][j]
            a1.append("-")
            a2.append(seq2[j - 1])
            j -= 1
            state = prev
        else:
            break

    return best, "".join(reversed(a1)), "".join(reversed(a2))

def _last_row_nw(seq1: str, seq2: str, match=2, mismatch=-1, gap=-2) -> List[int]:
    prev = [j * gap for j in range(len(seq2) + 1)]
    for i in range(1, len(seq1) + 1):
        curr = [i * gap] + [0] * len(seq2)
        for j in range(1, len(seq2) + 1):
            curr[j] = max(
                prev[j - 1] + score(seq1[i - 1], seq2[j - 1], match, mismatch),
                prev[j] + gap,
                curr[j - 1] + gap
            )
        prev = curr
    return prev

def hirschberg(seq1: str, seq2: str, match=2, mismatch=-1, gap=-2) -> Tuple[int, str, str]:
    def rec(a: str, b: str) -> Tuple[str, str]:
        if not a:
            return "-" * len(b), b
        if not b:
            return a, "-" * len(a)
        if len(a) == 1 or len(b) == 1:
            _, x, y = needleman_wunsch(a, b, match, mismatch, gap)
            return x, y

        mid = len(a) // 2
        left = _last_row_nw(a[:mid], b, match, mismatch, gap)
        right = _last_row_nw(a[mid:][::-1], b[::-1], match, mismatch, gap)

        split, best = 0, None
        for j in range(len(b) + 1):
            val = left[j] + right[len(b) - j]
            if best is None or val > best:
                best = val
                split = j

        l1, l2 = rec(a[:mid], b[:split])
        r1, r2 = rec(a[mid:], b[split:])
        return l1 + r1, l2 + r2

    a1, a2 = rec(seq1, seq2)
    return pairwise_alignment_score(a1, a2, match, mismatch, gap), a1, a2

def blast_style(seq1: str, seq2: str, k=3, threshold=6, match=2, mismatch=-1) -> Tuple[int, str, str]:
    index = {}
    for j in range(len(seq2) - k + 1):
        mer = seq2[j:j + k]
        index.setdefault(mer, []).append(j)

    best = (0, "", "")
    for i in range(len(seq1) - k + 1):
        mer = seq1[i:i + k]
        if mer not in index:
            continue
        for j in index[mer]:
            left_i, left_j = i, j
            while left_i > 0 and left_j > 0 and seq1[left_i - 1] == seq2[left_j - 1]:
                left_i -= 1
                left_j -= 1

            right_i, right_j = i + k, j + k
            while right_i < len(seq1) and right_j < len(seq2) and seq1[right_i] == seq2[right_j]:
                right_i += 1
                right_j += 1

            s1 = seq1[left_i:right_i]
            s2 = seq2[left_j:right_j]
            sc = sum(match if a == b else mismatch for a, b in zip(s1, s2))
            if sc >= threshold and sc > best[0]:
                best = (sc, s1, s2)

    return best

def _minimizers(seq: str, k=5, w=10) -> List[Tuple[str, int]]:
    if len(seq) < k:
        return []
    kmers = [(seq[i:i + k], i) for i in range(len(seq) - k + 1)]
    mins = []
    for start in range(max(1, len(kmers) - w + 1)):
        window = kmers[start:start + w]
        mins.append(min(window, key=lambda x: x[0]))
    out = []
    seen = set()
    for x in mins:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def minimizer_align(seq1: str, seq2: str, k=5, w=10) -> Dict:
    mins1 = _minimizers(seq1, k, w)
    mins2 = _minimizers(seq2, k, w)

    index2 = {}
    for mer, j in mins2:
        index2.setdefault(mer, []).append(j)

    anchors = []
    for mer, i in mins1:
        for j in index2.get(mer, []):
            anchors.append((i, j))

    anchors.sort()
    chain = []
    last_i, last_j = -1, -1
    for i, j in anchors:
        if i > last_i and j > last_j:
            chain.append((i, j))
            last_i, last_j = i, j

    return {"num_anchors": len(anchors), "chain_length": len(chain), "chain": chain}