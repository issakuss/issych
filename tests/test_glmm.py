import unittest
import numpy as np
import pandas as pd
import pingouin as pg

from issych.glmm import GlmmTMB


def generate_data_cat_x_cat() -> pd.DataFrame:
    """
    group（intervention/control） x　time（pre/post） x trait（連続変数）のデータ
    """
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


def generate_data_seq_x_seq() -> pd.DataFrame:
    """
    - thoughtのperformanceに対する主効果なし
    - mood, acceptanceはperformanceに主効果あり
    - thoughtとmoodの交互作用がperformanceに影響する
    - acceptanceが高いほど、thoughtとmoodの交互作用効果が小さくなる
    - acceptanceは一人一回、ほかは複数回取得
    """
    np.random.seed(0)

    N_SUB = 100
    N_OBS_PER_SUB = 20

    BETA_MOOD = 1.0
    BETA_ACCEPTANCE = -0.8
    BETA_INTERACT = 1.2
    BETA_MOD = -0.8

    rows = []
    for sub_id in range(N_SUB):
        acceptance = np.random.normal(0, 1)

        for t in range(N_OBS_PER_SUB):
            thought = np.random.normal(0, 1)
            mood = np.random.normal(0, 1)

            interaction = thought * mood
            modulated_interaction = interaction * (1 + BETA_MOD * acceptance)

            performance = (
                BETA_MOOD * mood +
                BETA_ACCEPTANCE * acceptance +
                BETA_INTERACT * modulated_interaction +
                np.random.normal(0, 1))

            rows.append({
                "sub_id": sub_id,
                "time": t,
                "acceptance": acceptance,
                "thought": thought,
                "mood": mood,
                "performance": performance})

    df = pd.DataFrame(rows)
    return df


class TestGlmmTMB(unittest.TestCase):
    def test_cat_x_cat(self):
        data = generate_data_cat_x_cat()
        formula = 'score ~ group * time + (1 | sub_id)'
        model = GlmmTMB(data, formula).fit()

        result = model.summary()
        self.assertTrue(result.iloc[-1, -1] < .001)

        d = abs((result.est / model.sigma()).iloc[-1])
        self.assertTrue(d > 0.5)

        formula = 'score ~ group * time * trait + (1 | sub_id)'
        model = GlmmTMB(data, formula).fit()

        _ = model.emtrends(specs='group * time', var='trait')
        contrast = model.contrast(method='revpairwise')
        self.assertTrue(contrast.loc['5', 'p'] < .05)

    def test_seq_x_seq(self):
        data = generate_data_seq_x_seq()
        data['interact'] = data.thought * data.mood
        formula = 'performance ~ interact * acceptance + (1 | sub_id)'
        model = GlmmTMB(data, formula).fit()
        model.emtrends(
            specs='acceptance', var='interact',
            at={'acceptance': {'low': -1, 'mid': 0, 'high': 2}})
        contrast = model.contrast()
        self.assertTrue(contrast.loc['1', 'p'] < .05)
