# Robust Variational Inference via Credal Sets

This repository contains code and notes for preliminary experiments on credal mixtures for flow based variational inference under prior uncertainty.

## Notes
- Preliminary research note (PDF): ./preliminary_research_note.pdf
- Credal flow matching note (PDF): ./credal_flow_matching_note.pdf

## Core idea
Given K approximate posteriors q_k learned under different priors, we fit mixture weights w on the simplex by minimising held out negative log likelihood, and compare the optimised mixture to the best single component across random seeds.

## Reproducibility
### Setup
Create an environment and install requirements.

### Run experiments
- Train flows and produce checkpoints: `bash run_all.sh`
- Optimise mixture weights: `bash credal_run.sh`
- Evaluate held out NLL: `python evaluation_nll.py`

## Results
See the summary table in the preliminary research note.
Optionally export results to: `results/nll_summary.csv`

## Repository layout
- train_flow.py: training code
- opt_credal_kl.py: mixture weight optimisation
- evaluation_nll.py: held out NLL evaluation
- datasets/: synthetic datasets
- priors/: prior configurations
