# Code organization

The maintained PPA workflow trains a compatible numeric sequence model from RTL and constraints. Its training entrypoint shares the retained metric/decoding implementation with standalone evaluation. The public launcher requires explicit configuration and artifact locations, so each user defines an experiment rather than inheriting a private model or dataset identity.

The backend under `src/text2text_regression/` adapts Google DeepMind's regress-lm. RTL formatting, dataset wrappers, multi-metric target construction, Hugging Face training, sampled evaluation and experimental objectives are local research modifications. Google notices remain on adapted backend files. The exact imported upstream revision was not recorded.

`src/revolution/` retains the REvolution-derived population-search engine and its internal module name for import compatibility. It evolves candidate implementations, evaluates correctness, and ranks PPA using the selected proxy or synthesis evaluator. Explicit external benchmark, tool-flow and library locations replace installation-specific defaults.

`src/revolution/evolve/` contains the separate local EvolVE implementation and benchmark adapters. It was developed from paper inspiration and is maintained here. It is not a certified reproduction; the external benchmark source, generation model, verification strategy and scoring configuration determine any future comparison.

`experiments/training/` holds selected historical learning-objective attempts. These preserve implemented ideas with separate parsers and helper modules; they are not interchangeable validated defaults. The [experiment index](EXPERIMENTS.md) describes the evidence boundary.

The `results/v6.3/` tables and `figures/` plots are sanitized aggregates of recorded standalone model measurements. They contain no design-level records, checkpoints or raw datasets. The [results document](V63_RESULTS.md) explains their provenance and limitations.
