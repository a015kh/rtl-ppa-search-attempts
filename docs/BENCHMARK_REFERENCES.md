# RTL-OPT and Pluto benchmark references

The retained `test_rtl_opt` and `test_pluto` records are selected local DC/PT
remeasurements of implementations from these benchmark projects. They are not
complete official benchmark releases and do not reproduce the papers' original
PPA labels, synthesis environments or reported outcomes. The original imported
upstream commits were not recorded.

| Retained split | Local measured selection | Benchmark origin |
|---|---|---|
| `test_rtl_opt` | 48 distinct RTL variants, each at six clock targets: 288 rows; the saved model evaluation decoded 287 | [RTL-OPT authors' repository](https://github.com/hkust-zhiyao/RTL-OPT) and [paper](https://arxiv.org/abs/2601.01765) |
| `test_pluto` | Four example designs, three variants per design, six clock targets: 72 rows; all decoded | [Pluto authors' repository](https://github.com/scale-lab/Pluto) and [paper](https://arxiv.org/abs/2510.14756) |

The six recorded clock-target values are `1, 3, 5, 8, 10, 15`. These are repeated
measurement settings, not additional unique benchmark tasks. Pluto's selected
designs are ALU, FSM, Conway's Game of Life and a shift-register design, with
original, area/power-optimized and delay-optimized variants. The selection is
consistent with the four example families in the [paper appendix](https://arxiv.org/html/2510.14756v1).
Original generator references, retained record structure and public upstream
inventories corroborate the benchmark identities, but do not pin the historical
imported commits or establish complete selection/transform histories.

Use the retained counts above when describing this project's evaluation. The
RTL-OPT arXiv abstract describes 36 designs while the inspected repository README
describes 40; neither number is a substitute for the local 48-variant selection.
Likewise, the 72 Pluto measurement rows are not a claim about the size or coverage
of the full Pluto benchmark.

## References

**RTL-OPT:** Yao Lu, Shang Liu, Hangan Zhou, Wenji Fang, Qijun Zhang and Zhiyao Xie.
*A New Benchmark for the Appropriate Evaluation of RTL Code Optimization.*
arXiv:2601.01765, 2026. [Verified paper](https://arxiv.org/abs/2601.01765),
[DOI](https://doi.org/10.48550/arXiv.2601.01765). A proceedings venue was not
independently verified for this release.

**Pluto:** Manar Abdelatty, Maryam Nouh, Jacob K. Rosenstein and Sherief Reda.
*Pluto: A Benchmark for Evaluating Efficiency of LLM-generated Hardware Code.*
arXiv:2510.14756, 2025. [Verified preprint](https://arxiv.org/abs/2510.14756),
[DOI](https://doi.org/10.48550/arXiv.2510.14756). The authors' [publication list](https://scale-lab.github.io/pubs/)
also identifies a 2026 IEEE International Conference on LLM-Aided Design (LAD)
publication. Its proceedings DOI was not recovered. The citation below identifies
the verified **2025 arXiv version**; its year should not be silently changed to
the later conference year.

Machine-readable citations are in [references.bib](../references.bib).
Benchmark code/data licenses are separate from this repository's source licenses;
these citations do not grant redistribution rights. Benchmark payloads and local
per-design measurements remain excluded from the public repository.
