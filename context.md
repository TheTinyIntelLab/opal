# The Tiny Intelligence Project

The Tiny Intelligence Project is an independent, solo research lab asking **how much useful capability can be built by representing a problem's structure explicitly?**

The lab studies specialized scientific problems where mathematical structure can improve prediction, generalization, or experimental decisions. Its method is to identify that structure, build a model around it, and compare it with credible alternatives under a reproducible protocol.

"Tiny" describes an ambition for efficiency, not a predetermined parameter ceiling. The right model is the one with enough capacity for the problem and evidence that its capacity helps. A larger model is justified when it delivers a reproducible improvement. A smaller model is preferable when its accuracy, reliability, and coverage are comparable. Both decisions require measurements.

The lab's first project is Opal. The deliverable is a complete research result and a usable model: a tested codebase, reproducible experiments, downloadable artifacts, and a runnable Hugging Face demonstration. An attractive demo cannot substitute for scientific evidence, and a research result should be possible for another person to run.

## Opal v1

Opal is a model family for biological interventions. **Versioning starts at Opal v1.** Its first task is predicting the mean transcriptomic response to an unmeasured pair of perturbations when the individual perturbations have already been measured in the same biological context.

For interventions A and B, define their effects relative to matched controls as delta_A and delta_B. The combined effect is decomposed as

$$
\delta_{AB}=\delta_A+\delta_B+\epsilon_{AB}.
$$

Opal uses the observed individual effects as an explicit additive prediction and learns a correction:

$$
\boxed{
\hat\delta_{AB}=\delta_A+\delta_B+f_\theta(A,B)
}
$$

with

$$
f_\theta(A,B)=f_\theta(B,A).
$$

The symmetry expresses the experimental task: simultaneous unordered interventions. It does not describe sequential treatments.

The interaction residual is defined on a specified expression scale. Additivity in log-transformed expression differs from additivity in normalized counts. The observed single profiles are estimates, so their measurement error also enters the residual. The residual measures departure from the chosen prediction baseline; it is not automatically a causal mechanism, a fitness interaction, or drug synergy.

The primary task is often called seen 2/2 combination prediction. Both components are available individually, while their combination is withheld from model fitting. Predicting an unseen component, another cell type, a different dose, or an individual cell's trajectory requires additional evidence. Opal v1 will state its supported contexts and inputs precisely.

## The scientific question

> Can an explicit additive prediction, symmetric pair features, and a shared low-dimensional residual representation improve prediction of unmeasured combinations relative to strong classical and neural alternatives?

This question contains separable hypotheses:

- An explicit additive term can improve the behavior of a learned predictor.
- Symmetric multiplicative features can capture useful pair structure.
- A limited number of shared output directions can improve generalization.
- Representations learned from singles may transfer to combination prediction.
- These choices may improve accuracy per training pair and per unit of compute.

None is guaranteed. The experiments distinguish their contributions rather than treating the entire architecture as one unexplained package.

Residual MSE equals combined-response MSE when the same observed additive term and evaluation genes are used. Residual training therefore changes the parameterization and possible regularization of the predictor, not the squared-error objective by itself. Opal's evidence must come from held-out performance, controlled ablations, and recovery of reproducible residual signal.

## Architecture and capacity

Opal v1 has one fixed output-level decomposition and a bounded set of candidate interaction models. The research will compare a coordinate-product model, learned symmetric bilinear features, and a more expressive symmetric residual network. The released architecture will be chosen using development data before final test evaluation.

The original coordinate-product candidate is

$$
e_u=E_{\mathrm{single}}(\delta_u),
\qquad
\hat\epsilon_{AB}=b+B\Lambda^\top(e_A\odot e_B).
$$

Here, e_u has dimension m, B has shape G by k, and Lambda has shape m by k. The residual lies in an affine output space of dimension at most k. The optional shared offset b is explicitly compared with a training-mean residual baseline.

A stronger candidate learns the coordinates in which multiplication occurs:

$$
q_{AB}=(Pe_A)\odot(Pe_B),
\qquad
\hat\epsilon_{AB}=b+B\Lambda^\top q_{AB}.
$$

This preserves pair symmetry while allowing interactions between different original embedding coordinates. A symmetric residual network tests whether further nonlinear capacity is useful. Embedding dimension, interaction-feature dimension, and output rank are separate quantities.

Single-effect representations can be fitted by PCA or a denoising encoder trained on single-condition measurements. Frozen representations provide a controlled reference. Joint fine-tuning is a required comparison and can become the released configuration if it generalizes better. Freezing reduces one route to fitting training pairs; it does not prevent overfitting or establish a biological interpretation.

Large latent transition matrices are not the default for a single fixed context. An unconstrained matrix acting on only one fixed state is observationally equivalent to a vector, and a nonlinear decoder does not preserve output-level additivity. State-dependent operators become meaningful when the data and evaluation support varying states, or when additional constraints give those matrices a testable purpose.

Capacity is selected from validation accuracy, stability, sample efficiency, memory, and latency. The lab will report trainable parameters, all inference parameters, stored reference profiles, and upstream resources separately. It will not call a system tiny solely because one trainable adapter is small.

## Evidence for v1

Norman 2019 is the first implementation dataset. It is a K562 CRISPR activation screen with single and paired interventions, not a knockout experiment. Its processed variants must be identified separately from its original data. [Norman et al., 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6746554/)

Norman alone is insufficient for the intended claim. A recent benchmark recommends broader combinatorial evaluation because additive prediction is difficult to surpass there. Opal v1 therefore includes an independent genetic benchmark, with Wessels CaRPool-seq as the planned second dataset. Its Cas13 RNA-targeting experiment also tests the method outside CRISPR activation. [Miller et al., 2026](https://www.nature.com/articles/s41587-026-03307-w), [Wessels et al., 2023](https://pubmed.ncbi.nlm.nih.gov/36550277/)

ComboSciPlex is the planned chemical validation dataset. It tests whether the architecture and training procedure remain useful after being fitted to drug-combination data. This is method transfer, not a claim that Norman-trained weights predict drugs without training. [CPA study](https://pmc.ncbi.nlm.nih.gov/articles/PMC10258562/)

The benchmark suite includes:

- Pure additive, no-change, training-mean response, and training-mean residual predictors, which establish how much requires pair-specific learning.
- Regularized symmetric regression and a global nonlinear additive-response correction, which test simpler explanations of apparent interactions.
- Capacity-matched symmetric neural models with the same inputs, which test the proposed structural choices.
- Public methods including SLIM, CPA, GEARS, and PertAdapt, with their input access and upstream resources disclosed.
- NAIAD and related combination predictors in the novelty review, with direct comparisons only when their output task is compatible.

The comparisons must separate architecture effects from access to observed singles, pretrained representations, and external biological knowledge. Published scores under another preprocessing or split are context, not head-to-head results.

## Evaluation and reliability

The primary assessment combines condition-weighted expression error with improvement over additive prediction. It includes a full frozen gene universe, a component-informed panel derived from observed singles, residual alignment, and evaluation of emergent genes under a prespecified evaluator-only diagnostic.

An advantage-versus-interaction-magnitude plot is useful, but it cannot carry the whole claim. Larger residuals give the additive baseline more room to improve. The analysis includes normalized skill, stronger comparators, shuffled-target controls, and measurement-reliability diagnostics.

Cells estimate a condition's mean response; they do not become independent perturbation experiments. Uncertainty must account for shared single profiles, shared controls, training randomness, and any genuine experimental replicates available in the source. Split-half profiles provide a measurement-reliability reference, not a universal upper bound on model accuracy.

The code will separate allowed fitting inputs from held-out combination responses. Source provenance, canonical perturbation identities, gene order, preprocessing parameters, split manifests, and run configurations will be saved with the results.

## A complete first release

Opal v1 is complete when the research protocol has been executed, its conclusions are supported by the recorded evidence, and the artifacts run independently of the researcher's machine. Required work includes:

1. A verified data and evaluation pipeline across the benchmark datasets.
2. Strong baseline comparisons, structural ablations, capacity and sample-efficiency studies, and a harder combination holdout.
3. A selected architecture with measured CPU inference behavior and an assessed reliability signal.
4. A tested Python package, reproducible commands, release checkpoints, and immutable evaluation artifacts.
5. A Hugging Face model repository and working Space showing the additive prediction, learned correction, supported context, and downloadable results.
6. A research report and model card stating what succeeded, what failed, and what the evidence does not establish.

A model's performance cannot be promised by a roadmap. Weak or inconclusive results must be published honestly, with their consequences for the hypotheses. The lab will not declare a structural advance merely to satisfy a release name.

Future Opal versions can add genuinely new capability, such as context generalization, unmeasured single perturbations, dose dependence, higher-order combinations, or experimentally validated intervention selection. Routine fixes belong in patch releases. Other scientific domains can become separate model families when their questions and evaluation differ.

The detailed implementation and research decisions live in [roadmap.md](roadmap.md). This file defines the lab's purpose, Opal v1's scope, and the standard its first release must meet.
