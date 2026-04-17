1. **Fix missing `np` import in `field_observation_spectral_kwargs`**
   - The test `test_bandpass_from_transmission_file` failed because `field_observation_spectral_kwargs` in `src/torchsky/filter_profiles/inr_bandpass.py` references `np.asarray` but `import numpy as np` was removed from the file's top level.
   - I need to add `import numpy as np` inside `field_observation_spectral_kwargs` just like I did for the other functions.
2. **Complete pre commit steps**
   - Run tests and linting to ensure it's fully fixed.
3. **Submit the change**
   - Submit the PR.
