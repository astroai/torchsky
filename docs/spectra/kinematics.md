# Kinematics
<!-- TODO(0.5.0 docs): intro (redshift, relativistic Doppler, velocity conventions and frames) -->

## Redshift Helpers

Sample-wise helpers live in `torchsky.redshift` and are re-exported here.
`apply_redshift` shifts a spectrum by an explicit velocity kind;
`shift_wavelength`, `dimming_f_lambda` (\(F_\lambda\) densities, \(1/(1+z)\)), and
`dimming_tolman_sb` (surface brightness, \((1+z)^{-4}\)) are different
operations, so do not mix them. Rapidity helpers (`zeta_from_redshift`,
`compose_rapidities`, `shift_ln_wavelength`) and `los_barycentric_beta` follow
\(\ln\lambda_\mathrm{topo} = \ln\lambda_\mathrm{rest} + \zeta_\mathrm{sys} + \zeta_\mathrm{BERV}\)
with \(v_\mathrm{BERV} > 0\) when the observer recedes. LOSVD and published-\(R\)
line-spread convolution share `convolve_log_lambda` and `sigma_R_kms`
(\(\sigma = c / (2.355\,R)\)).
