# Opal next-cycle model card

This is a local research preview. The code implements observed-single and separately identified estimated-single prediction, but biological context-transfer confirmation and reliable uncertainty under context shift are not established.

## Supported inputs and outputs

The observed-single checkpoint requires registered contexts, exact intervention/dose/time identities, matching assay and modality, both measured single effects, and a fixed gene order. It predicts a population-mean pair effect relative to the matched vehicle and exposes the additive and residual terms. Control and single routes return explicit references. Unknown contexts, incompatible genes, repeated targets and unavailable singles are rejected.

The estimated-single checkpoint has its own `opal-contextual-estimated-singles-v1` manifest. Each single is identified as measured or estimated, with its source contexts. The imputer can use the same exact intervention measured in other contexts; it abstains if that reference does not exist. Its outputs have no calibrated interval, and the measured-availability mask never changes.

The native CPA export supports registered exact-dose chemical pairs with known context categories. It stores weights and permitted references, contains no raw combination outcomes, and restores the original predictions exactly. It requires the separate Python 3.10/CUDA native environment. `tools/predict_native_cpa.py` validates fingerprints before predicting.

## Data and exposure

The exploratory EmeraldBay subset uses six pinned public shards, eight cell lines, 17 exact-dose pair conditions and 1,024 genes selected from vehicle controls. There are 54 fitting, 18 calibration and 27 evaluator conditions. The held cell lines are NCI-H1573 and SW1417. Query-context controls and measured singles are allowed adaptation resources. This pilot is already exposed.

The D-SPIN procedure test uses a separately downloaded, checksum-verified study with 10 lineage/batch contexts and 1,011 genes intersected with the previously frozen gene list. Myeloid and NK lineages are held out across both batches. Published lineage labels were assigned after treatment, batches are not biological replicates, and dose pairs share four compound identities. The frozen kernel procedure is refitted only on the new source-context fitting conditions; EmeraldBay fitted weights are not pooled across assays.

Historical Norman/Wessels evaluator outcomes were not reopened. The native GEARS adaptation is evaluated on development folds with the original architecture and loss, using aggregate observations and graphs derived only from permitted single profiles. This is not canonical GEARS with external ontology and individual cells. CPA is similarly an aggregate-input adaptation of its native implementation.

## Results and restrictions

EmeraldBay MSE is 0.210239 for additive, 0.172218 for the selected kernel and 0.157098 for native CPA. These comparisons are retrospective. D-SPIN MSE is 0.034025 for the frozen procedure versus 0.033108 for the source-selected reference, so the independent material-skill gate fails. The nominal 90% empirical source radius covers only 34.3% of held-lineage conditions; it must not be presented as a calibrated OOD interval.

Missing-single source validation improves only 0.5% over averaging a target across contexts. Its pilot total MSE is 0.092829, but the main-effect and residual errors have a large negative cross term. The result is diagnostic, not evidence that estimated singles transfer reliably or that a learned encoder is justified.

The default full-gene PertAdapt adapter exceeds the RTX 4060's 8 GB memory in the recorded feasibility probe. Its distributed launcher also requires NCCL, unavailable in this Windows build. A compatible complete native run and verified full foundation gene alignment remain pending.

Higher-order mixtures, sequential treatment, unseen intervention identities without measured sources, single-cell distributions, mechanistic explanations and broad transfer claims are outside this release.

## Resources and terms

The observed-single kernel pilot export is about 1.7 MB including its reference bank. Earlier cold-process profiling measured about 65 MB peak working set and roughly 6 ms median single-pair latency with one numerical thread. Those figures apply to this pilot only. Native CPA training allocated about 33 MB peak CUDA tensor memory; GEARS development runs allocated about 150 MB. These figures exclude driver overhead and CPU process memory. Exact artifacts and hashes are in `results/next/research-ledger-v2`.

EmeraldBay is CC BY 4.0. CPA source is BSD 3-Clause and GEARS source is MIT; the provenance bundles retain their notices. Dataset, backbone and graph terms remain distinct from the repository license. The supplied PertAdapt resources and D-SPIN derivatives have not been cleared for a public redistribution bundle and stay local. No public upload was performed, and the user-requested master plan is excluded from Git and source archives.

## Reproduction

See `docs/next_cycle.md` for preparation, native environments, evaluation and inference commands. The frozen independent procedure config is `configs/datasets/dspin-confirmation-v1.json`. Choose a new output directory for every run; reproducing an already evaluated dataset does not create new confirmation evidence.
