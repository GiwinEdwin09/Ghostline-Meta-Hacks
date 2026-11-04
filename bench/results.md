# Ghostline benchmark results

Generated 2026-11-04 07:19 UTC, back end in live mode, prompts 2918bf51e83c, API 0.1.0.

**Our 95% intervals contained the true value in 4 of 15 clips (27%; 95% Wilson interval for that share 11% to 52%).** Target: 90 to 98%.
The median absolute error was 7.8%, and 5 of 12 measurements agreed with the truth within their combined uncertainty (|z| ≤ 1.96).
The right physics was chosen in 10 of 15 clips where it is defined (67%).
The deliberately bad clips: 1 of 3 came out low confidence for the right reason.

| Clip | System | Quantity | Truth | Measured [95% interval] | Error | z | In interval | Model | Confidence |
|---|---|---|---|---|---|---|---|---|---|
| FF-G1 | Free fall | g (m/s^2) | 9.810 ± 0.010 | 11.37 [11.11, 11.63] | +15.9% | 12 | no | FF0 ✓ | low |
| FF-G4 | Free fall | g (m/s^2) | 9.810 ± 0.010 | 11.25 [11.04, 11.46] | +14.7% | 13 | no | FF0 ✓ | low |
| FF-G5 | Free fall | g (m/s^2) | 9.810 ± 0.010 | 11.49 [11.21, 11.77] | +17.1% | 12 | no | FF0 ✓ | low |
| FF-G6 | Free fall | g (m/s^2) | 9.810 ± 0.010 | 11.01 [10.78, 11.24] | +12.2% | 10 | no | FF0 ✓ | low |
| FF-F7 | Free fall | g (m/s^2) | 9.810 ± 0.010 | 12.44 [12.16, 12.72] | +26.8% | 18 | no | FF0 ✗ (expected FF1|FF2) | low |
| FF-F8 | Free fall | g (m/s^2) | 9.810 ± 0.010 | 11.29 [11.06, 11.51] | +15.0% | 13 | no | FF0 ✗ (expected FF1|FF2) | low |
| PD-2 | Pendulum | g (m/s^2) | 9.810 ± 0.010 | 9.969 [9.811, 10.13] | +1.6% | 2.0 | no | P2 ✓ | medium |
| PD-9 | Pendulum | g (m/s^2) | 9.810 ± 0.010 | 10.14 [9.979, 10.30] | +3.3% | 4.0 | no | P2 ✓ | low |
| PD-10 | Pendulum | g (m/s^2) | 9.810 ± 0.010 | no result | – | – | – | – ✗ (expected P2|P3|P4) | – |
| PD-11 | Pendulum | g (m/s^2) | 9.810 ± 0.010 | 9.793 [9.637, 9.949] | -0.2% | -0.21 | yes | P2 ✓ | low |
| PD-12 | Pendulum | g (m/s^2) | 9.810 ± 0.010 | no result | – | – | – | – ✗ (expected P2|P3|P4) | – |
| PD-26 | Pendulum | g (m/s^2) | 9.810 ± 0.010 | 9.831 [9.674, 9.988] | +0.2% | 0.26 | yes | P2 ✓ | medium |
| BN-18 | Bounce | e | 0.8900 ± 0.010 | 0.8802 [0.8398, 0.9207] | -1.1% | -0.43 | yes | B0 ✓ | low |
| BN-19 | Bounce | e | 0.8900 ± 0.010 | no result | – | – | – | – ✗ (expected B0) | – |
| BN-20 | Bounce | e | 0.8900 ± 0.010 | 0.8812 [0.8498, 0.9126] | -1.0% | -0.47 | yes | B0 ✓ | low |

## Deliberately bad clips

Expected: low confidence, for the reason the clip was made to show.

| Clip | What is wrong | Confidence | Reasons Ghostline gave | Flagged correctly |
|---|---|---|---|---|
| X-23 | camera motion | no result | camera_motion | yes |
| X-24 | out of plane | low | camera_motion, few_periods | no |
| X-25 | few periods | low | out_of_plane, camera_motion | no |

## By system

| System | Clips | Contain truth | Median abs. error |
|---|---|---|---|
| Bounce | 3 | 2 of 3 | 1.0% |
| Free fall | 6 | 0 of 6 | 15.5% |
| Pendulum | 6 | 2 of 6 | 0.9% |

## Not scored

- PD-10: failed. This experiment reached its model budget before it finished.
- PD-12: failed. The model isn't available to this API key. Check the model names in the settings.
- BN-19: failed. The camera moved 7.0 px during the clip, which is too much to measure. Film again with the phone on a tripod.
- SD-3: failed. trajectory.parquet has no column(s) ['phi'] for a spin_down clip.
- SD-22: failed. trajectory.parquet has no column(s) ['phi'] for a spin_down clip.

## How the truth was obtained

- **g** (free fall, pendulum): local gravity from latitude and altitude, 9.780327 (1 + 0.0053024 sin²φ − 0.0000058 sin²2φ) − 3.086×10⁻⁶ h.
- **μ_k** (ramp, block): tan θ of the tilt at which the block slides at constant speed; five repeats, mean ± SD.
- **β** (ramp, ball): theory for a solid sphere, β = 2/5, so a = (5/7) g sin α.
- **e** (bounce): the click timing in a normal-speed recording, e = Δt(n+1) / Δt(n), averaged over the drops.
- z = (measured − truth) / √(σ_measured² + σ_truth²), with σ_measured taken from the 95% interval.

Model spend for this pass: $13.96.
