import unittest
import numpy as np
import pandas as pd
import pingouin as pg

from issych.glmm import GlmmTMB


def generate_data() -> pd.DataFrame:
    np.random.seed(0)

    N_SUB = 100
    N_OBS_PER_PHASE = 20  # pre:20, post:20
    RATE_MISS = 0.1
    TRAIT_EFFECT_MODIFIER = -0.5

    n_obs = N_OBS_PER_PHASE * 2
    sub_id = np.repeat(np.arange(N_SUB), n_obs)
    group = np.repeat(np.where(np.arange(N_SUB) < (N_SUB / 2),
                               'control', 'intervention'), n_obs)
    time = np.tile(np.repeat(['pre', 'post'], N_OBS_PER_PHASE), N_SUB)

    trait_per_sub = np.random.normal(loc=0, scale=1, size=N_SUB)
    trait = np.repeat(trait_per_sub, n_obs)

    random_intercepts = np.random.normal(loc=0, scale=5, size=N_SUB)
    random_intercepts_full = np.repeat(random_intercepts, n_obs)

    base_effect = np.where((group == 'intervention') & (time == 'post'), 1, 0)
    modulated_effect = base_effect * (5 + TRAIT_EFFECT_MODIFIER * trait)

    score = (
        50
        + modulated_effect
        + random_intercepts_full
        + np.random.normal(loc=0, scale=10, size=N_SUB * n_obs)
    )

    n_total = len(score)
    n_missing = int(n_total * RATE_MISS)
    ids_missing = np.random.choice(n_total, size=n_missing, replace=False)
    score[ids_missing] = np.nan

    data = pd.DataFrame({
        'sub_id': sub_id,
        'group': group,
        'time': time,
        'trait': trait,
        'score': score
    })

    return data


class TestGlmmTMB(unittest.TestCase):
    def test(self):
        data = generate_data()
        formula = 'score ~ group * time + (1 | sub_id)'
        model = GlmmTMB(data, formula).fit()

        result = model.summary()
        self.assertTrue(result.iloc[-1, -1] < .001)

        d = abs((result.coef / model.sigma()).iloc[-1])
        self.assertTrue(d > 0.5)

        formula = 'score ~ group * time * trait + (1 | sub_id)'
        model = GlmmTMB(data, formula).fit()
        trends = model.contrast(on='group * time', compareby='trait')
        self.assertTrue(trends.p < .05)
