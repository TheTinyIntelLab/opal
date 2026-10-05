# Architecture experiments

These are exploratory development studies conducted after the historical outcomes were known. All scores weight validation occurrences equally; partner folds overlap.

Reproduce with `python main.py frontier`. Completed experiment directories are preserved rather than overwritten.

## Measured outcomes

Adding eligible protected Wessels training pairs reduces same-population MSE by 16.05% on random folds and 4.92% on partner folds. Adding variance weighting gives 18.73% and 4.91%. This comparison fixes the original validation identities and gene panel, and excludes historical query identities from training.

The baseline residual kernel is selected in every competitive architecture outer fold. GP, residual retrieval and the fixed four-estimator, eight-output-PC TabICLv2 comparator do not beat it on the original datasets. Semantic priors have small, inconsistent gains; shuffled identity controls must be considered.

- Norman random: measured-variance gain 1.27% to 2.54%; permuted control -0.06% to 0.19% across three dependent split seeds.
- Norman partner: measured-variance gain -0.58% to 1.13%; permuted control 0.00% to 0.00% across three dependent split seeds.
- Wessels original random: measured-variance gain 0.21% to 1.05%; permuted control -0.08% to 0.10% across three dependent split seeds.
- Wessels original partner: measured-variance gain -0.07% to 0.19%; permuted control -0.01% to 0.00% across three dependent split seeds.
- Wessels protected random: measured-variance gain 0.92% to 1.24%; permuted control -0.05% to 0.02% across three dependent split seeds.
- Wessels protected partner: measured-variance gain -0.40% to 0.05%; permuted control -0.01% to -0.00% across three dependent split seeds.

## Exposure and limits

The broad expansion moves 12 original query identities into development. The protected expansion moves zero, but 30 of its query identities had already appeared in the broad development track. Neither expansion is independent confirmation.

Joint technical bootstraps share single and control draws across conditions. The noise likelihood uses diagonal pair variances and does not model their full covariance. GP intervals exclude query measurement variance and selection uncertainty. No calibrated abstention rule is claimed.

External transfer uses independent K562 CRISPRi single effects on the released X scale to initialize a GO encoder and output directions. It is representation transfer, with a separately fitted current-context pair head. The external scale is not equated with Opal log2 pseudobulk effects.

PRESAGE includes semantic priors and DEG rankings, not the required expression matrices; expression was separately downloaded from scPerturb record 7278143 and verified by publisher MD5. Semantic derivatives have CC BY-NC 4.0 terms and are excluded from Git.

CPU-only runs leave native contextual PertAdapt/GEARS/CPA, broad hyperparameter tuning of foundation models, independent confirmation and frontier status unresolved. Every completed score, including unsuccessful additions, is retained in results/architecture/report.json.

## Transfer and objective controls

- Norman random: kernel=0.006292; linear_mse=0.008281; linear_weighted=0.008517; no_transfer=0.008892; selected_transfer=0.008281; transfer_both=0.008613; transfer_decoder=0.008895; transfer_encoder=0.008549.
- Norman partner: kernel=0.007181; linear_mse=0.009534; linear_weighted=0.009743; no_transfer=0.010484; selected_transfer=0.009834; transfer_both=0.010869; transfer_decoder=0.011279; transfer_encoder=0.010785.
- Wessels expanded random: kernel=0.006471; linear_mse=0.023141; linear_weighted=0.028550; no_transfer=0.010658; selected_transfer=0.010943; transfer_both=0.011237; transfer_decoder=0.010621; transfer_encoder=0.011229.
- Wessels expanded partner: kernel=0.014467; linear_mse=0.415128; linear_weighted=0.479667; no_transfer=0.019227; selected_transfer=0.030557; transfer_both=0.023492; transfer_decoder=0.019632; transfer_encoder=0.022255.
- Wessels protected random: kernel=0.007360; linear_mse=0.027270; linear_weighted=0.031739; no_transfer=0.012031; selected_transfer=0.012031; transfer_both=0.012512; transfer_decoder=0.012093; transfer_encoder=0.012413.
- Wessels protected partner: kernel=0.016063; linear_mse=2.346815; linear_weighted=2.903400; no_transfer=0.021088; selected_transfer=0.021478; transfer_both=0.023081; transfer_decoder=0.021576; transfer_encoder=0.022368.
- Wessels original random: kernel=0.005032; linear_mse=0.006618; linear_weighted=0.006678; no_transfer=0.007193; selected_transfer=0.006618; transfer_both=0.007398; transfer_decoder=0.007775; transfer_encoder=0.007259.
- Wessels original partner: kernel=0.009868; linear_mse=0.010943; linear_weighted=0.011011; no_transfer=0.016089; selected_transfer=0.010943; transfer_both=0.016447; transfer_decoder=0.016792; transfer_encoder=0.015757.

## Coverage split sensitivities

- random, extra_pairs: 16.05% to 17.60% MSE reduction across three dependent seeds.
- random, extra_pairs_noise: 18.73% to 19.54% MSE reduction across three dependent seeds.
- partner, extra_pairs: 4.00% to 4.99% MSE reduction across three dependent seeds.
- partner, extra_pairs_noise: 4.08% to 4.99% MSE reduction across three dependent seeds.

## Bounded-feature sensitivity

A late separate control clips training-standardized features at magnitude eight during fitting and prediction. It reduces the severe expanded-fold extrapolation errors but does not beat the kernel. Both unbounded results and interrupted optimization records remain preserved.

- Norman random: bounded ordinary MSE=0.008190; bounded weighted MSE=0.008413; kernel MSE=0.006292.
- Norman partner: bounded ordinary MSE=0.008810; bounded weighted MSE=0.008947; kernel MSE=0.007181.
- Wessels expanded random: bounded ordinary MSE=0.010730; bounded weighted MSE=0.011126; kernel MSE=0.006471.
- Wessels expanded partner: bounded ordinary MSE=0.017244; bounded weighted MSE=0.017678; kernel MSE=0.014467.
- Wessels protected random: bounded ordinary MSE=0.012413; bounded weighted MSE=0.012865; kernel MSE=0.007360.
- Wessels protected partner: bounded ordinary MSE=0.019325; bounded weighted MSE=0.019807; kernel MSE=0.016063.
- Wessels original random: bounded ordinary MSE=0.006615; bounded weighted MSE=0.006675; kernel MSE=0.005032.
- Wessels original partner: bounded ordinary MSE=0.010921; bounded weighted MSE=0.010985; kernel MSE=0.009868.
