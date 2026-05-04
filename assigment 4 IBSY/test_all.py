from pairwise import (
    needleman_wunsch,
    smith_waterman,
    gotoh,
    hirschberg,
    blast_style,
    minimizer_align,
)
from msa import progressive_msa, iterative_refinement_msa, SimpleProfileHMM

def test_needleman():
    score, a1, a2 = needleman_wunsch("GATTACA", "GCATGCU")
    assert isinstance(score, int)
    assert len(a1) == len(a2)

def test_smith_waterman():
    score, a1, a2 = smith_waterman("ACACACTA", "AGCACACA")
    assert score >= 0
    assert len(a1) == len(a2)

def test_gotoh():
    score, a1, a2 = gotoh("GATTACA", "GCATGCU")
    assert len(a1) == len(a2)

def test_hirschberg():
    score, a1, a2 = hirschberg("GATTACA", "GCATGCU")
    assert len(a1) == len(a2)

def test_blast_style():
    score, a1, a2 = blast_style("ACTGACTGACTG", "TTTACTGACTGAA", k=3)
    assert score >= 0

def test_minimizer():
    result = minimizer_align("ACTGACTGACTG", "TTTACTGACTGAA")
    assert "chain_length" in result

def test_progressive_msa():
    seqs = ["ACGT", "ACGA", "ACGG"]
    msa = progressive_msa(seqs)
    assert len(msa) == 3
    assert len(set(len(x) for x in msa)) == 1

def test_iterative_msa():
    seqs = ["ACGT", "ACGA", "ACGG"]
    msa = iterative_refinement_msa(seqs)
    assert len(msa) == 3
    assert len(set(len(x) for x in msa)) == 1

def test_profile_hmm():
    msa = ["ACGT", "ACGA", "ACGG"]
    hmm = SimpleProfileHMM(msa)
    score, aligned = hmm.viterbi_align("ACGT")
    assert isinstance(score, float)
    assert isinstance(aligned, str)