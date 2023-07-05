from typing import Optional


class Pvalue2SigMark:
    def __init__(self, thresholds: Optional[dict]=None, ns_comment: str=''):
        """
        thresholds: dict
            {p-value_threshold: 'significance_mark'}.
            Strictest threshold must be first.
        ns_comment: str
            Comment to be returned when non-significant.
        """
        DEFAULT = {0.01: '**', 0.05: '*', 0.10: '†'}
        self.thresholds = thresholds or DEFAULT
        self.ns_comment = ns_comment

    def __call__(self, pvalue: float):
        for thr, mark in self.thresholds.items():
            if pvalue < thr:
                return mark
        return self.ns_comment