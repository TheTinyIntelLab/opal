# Root review of first numerical execution

The first run reproduces the exact v2 maximum validation discrepancy,
1.245199382493567e-08. Identity through coordinates, distances and the RBF kernel
is exact. The first divergence is kernel@whitener, with raw maximum
3.045525368217594e-11. Holding the final projection fixed while swapping global
versus query-local features reproduces the entire failing discrepancy. Holding
features fixed and changing final addition grouping changes at most
1.1102230246251565e-16. This is a causal localization, not a claim that all BLAS
implementations behave identically. Small feature scales amplify the whitening
roundoff; the observed worst-row scales are around 1e-5, not necessarily the
1e-8 floor. Do not report the scale floor as the measured cause.

The fixed-order candidate misses the immutable historical vector by
4.660858310412408e-08. It fails the frozen gate. Historical singleton callers
differ by 5.103748923485085e-08. This rules out the tested candidate and any
claim that matching the historical query path establishes general invariance;
it does not prove no conceivable implementation could satisfy the target.

A/C should finish independent fixed-state regression tests and their causal
review. No further repair menu is authorized. N3 may replay this successful
diagnostic twice using unchanged code and inputs from the root-built installed
wheel, after the root delivery run; it is deterministic verification, not a
changed diagnostic retry or another biological attempt. Root will record
installed paths in root_install_ready.json when available. No A-owned build is
needed. C should explicitly endorse or object to the no-successor decision.

The defensible present decision is no scientific successor because no repair
has passed both required gates. Preserve the failed candidate as an engineering
negative result and leave the strongest-reference biological question open.
