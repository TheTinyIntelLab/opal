# Evidence needed after the development gates

The current development result does not justify larger learned state models or
routing. A stronger test needs overlapping exact interventions across biological
states with independently repeated cultures, and baseline controls that predict
responses beyond source-target averaging. Genetic combination confirmation needs
actual co-interventions, matched constituent singles and controls, not chemical
target annotations or barcode-derived context identities.

Use an initial shared panel of twelve exact interventions selected from source
single coverage and response diversity. Specify one perturbation modality,
reagents, dose and endpoint across four development states. Each independent
culture block contains randomized matched controls, all twelve singles and a
registered subset of the 66 unordered pairs. Balance every condition across
batches; do not confound one state or intervention with an assay batch. Record
culture, donor, reagent, efficacy, assay, batch and well identities before
expression analysis. Use three independently initiated cultures per state as an
initial variance-estimation pilot, not a sufficient confirmation sample size.

Reserve a sealed fifth state and independently initiated culture blocks for a
single frozen-procedure confirmation. Main-effect evaluation withholds all its
single outcomes while allowing its controls. Observed-single pair transfer is a
different contract that reveals those singles but keeps pairs sealed. A separate
globally held pair block supports joint pair/context transfer. A predeclared
intervention panel held from all response fitting and external pretraining is
required for identity transfer. Do not use response-derived lineage labels as
pretreatment states; if sorting is needed, sort before treatment and preserve
whole-population aggregation as the primary endpoint.

The primary confirmation endpoint is the mean paired difference in condition MSE
between the frozen selected model and strongest source-selected simple reference.
Require at least 5% relative improvement and a positive paired confidence bound
under the registered culture/state sampling unit. Estimate the SD `s` of these
paired block differences in the variance pilot. For a requested 95% precision
half-width `h = 0.025 * pilot_reference_MSE`, a first normal approximation is `n = ceil((1.96*s/h)^2)`
independent blocks; replace it with a hierarchical simulation that preserves
shared-control covariance, intervention dependence, whole-state holdouts and
model-selection variation. Register `h` relative to the pilot reference MSE,
freeze that analysis before opening confirmation, and expand cultures or states
rather than treating extra cells as independent blocks.

Determine cells per well from within-condition mean-estimation precision on the
registered gene-vector endpoint. Simulate count sampling and matched controls
using pilot wells until technical measurement uncertainty is small enough to
resolve `h`; there is no universal minimum cell count. Gene selection and scaling
use development only or a separately declared baseline-reference adaptation.
Risk fitting, threshold calibration and confirmation require separate biological
blocks. At nominal 90%, even a finite rank radius needs at least nine calibration
groups, and finiteness alone does not establish useful width or shifted coverage.

This is a design specification for a future authorized experiment,
not a commissioned study. No collaborators were contacted and no paid resources
or prospective assays were ordered.
