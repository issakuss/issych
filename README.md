# How to install

```
pip install git+https://github.com/issakuss/issych
```

# Tutorial (WIP)

Prepare configure file like this.

```ini:monshi.ini
[range]
    _meta = ('A', 'B')
    scale1 = ('C', 'G')
    scale2 = 'scale2'

[scale1]
    idx_reverse = [1, 2]
    min_plus_max = 6
    nanpolicy = 'ignore'
    average = True
    
[scale2]
    preprocess = 'np.nan if pd.isna(q) else int(q > 3)'
    idx_reverse = [1, 3, 4]
    min_plus_max = 1
    subscale = {
        'a': [-1, 3, 5, 7, 9],
        'b': [2, 4, 6, 8, 10],
        'total': 'all'}
```
