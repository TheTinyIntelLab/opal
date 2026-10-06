# State-conditioned response architecture

For matched baseline expression `y0`, Opal models `dA = yA - y0` and
`rAB = dAB - dA - dB`. Pair prediction is `mA + mB + R(A,B,c)`;
absolute prediction is `y0 + mA + mB + R(A,B,c)`. The baseline is added once.

The compact prototype uses fold-fitted baseline PCA followed by a small learned
state encoder. It accepts a new compatible baseline directly. An intervention
encoder combines separate identity, numeric dose/mask and source-effect paths;
a 16-wide head predicts coefficients in a source-fitted response basis, anchored
to the exact intervention's source mean. These are predictive coordinates,
without an identifiable latent biological transition or temporal interpretation.

The residual encoder uses symmetric sums, absolute differences and optionally
coordinate products. Baseline dependence enters the main and residual heads;
there is no additional unconstrained context adjustment. A staged residual is
trained against genuine measured constituent effects. Joint training includes
single-response supervision to expose changes in the main head. A better pair
score does not override a failed single-response comparison.

Measured inputs are matched-control-relative effects supplied through an
explicit dictionary. Predicted exact interventions retain source support counts.
No denoising correction is applied. Descriptor-only unknown identities,
incompatible assays, reordered genes and repeated simultaneous targets are
rejected. Control and single routes bypass the pair residual.

The NumPy export persists gene order, assay, scale, exact intervention records,
control normalization, latent bases, all encoder/head weights and source support.
It contains no held response labels. New contexts require compatible processed
controls, not trained context categories. Reliability is experimental and
unavailable; state distance and intervention coverage are diagnostics. Single-cell
distributions, treatment-dependent composition forecasts and dynamics remain
outside the implemented prediction object.

Current two-seed exports also identify permitted source reference contexts,
biological state, study and assay for each main effect, and training condition
identities for the residual head. These are supporting observations, not causal
attribution. `opal.load_model(path)` dispatches the historical, contextual and
state schemas; each route retains its own input contract.
