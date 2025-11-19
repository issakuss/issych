Glossary
========

.. _issych-figure-setting-format:

issych-figure-setting-format
----------------------------

:mod:`issych.figure` で使用する、作図設定のフォーマットです。
:py:data:`matplotlib.rcParams` による、Figure 設定は、項目が多くて面倒です。
最低限の項目で、いい感じに設定するためのフォーマットです。

以下のような辞書で表されます。

.. code-block:: python

  {
      'color':
          main = '#EAE6E5',          # 主に用いられる色；黒など
          back = '#011627',          # 背景色；白など
          sub = '#3C4A57',           # mainで指定した色とは別の色；灰色など
          highlight = '#8FCB9B',     # 強調するための色；赤など
          subhighlight = '#EF6F6C',  # highlightで指定した色とは別の強調色；青など
          cycle = [                  # 複数の色が用いられる場合のパターン
              '#E69F00',             # 一つ目の色） 
              '#56B4E9',             # 二つ目の色）
              '#009E73',             # …色数は自由）
              '#F0E442',
              '#0072B2',
              '#D55E00',
              '#CC79A7'
          ],

      'size'
          figsize = [8.0, 4.0],      # [横幅, 縦幅] を表す数値
          fontsize = 16.0,           # フォントサイズを表す数値
          linewidth = 1.0,           # 線の太さを表す数値
          capsize = 1.0,             # エラーバーのキャップの長さを表す数値

      'misc'
          dpi = 256.0,               # 解像度を表す数値
          format = 'eps'             # 保存する際のフォーマットを表す文字列；pngなど
      
      'kwargs'                    # そのほか、rcParamsに直接渡したい設定項目
          'svg.fonttype' = 'none'    # 例：svgフォントの種類を指定
  }

``config/rcparams.toml`` というファイルにまとめておくことを推奨しています。
上記の例を ``toml`` ファイルにまとめると、以下のようになります。

.. code-block:: toml

  [color]
  main = '#EAE6E5'          # 主に用いられる色；黒など
  back = '#011627'          # 背景色；白など
  sub = '#3C4A57'           # mainで指定した色とは別の色；灰色など
  highlight = '#8FCB9B'     # 強調するための色；赤など
  subhighlight = '#EF6F6C'  # highlightで指定した色とは別の強調色；青など
  cycle = [                 # 複数の色が用いられる場合のパターン
    '#E69F00',              # 一つ目の色） 
    '#56B4E9',              # 二つ目の色）
    '#009E73',              # …色数は自由）
    '#F0E442',
    '#0072B2',
    '#D55E00',
    '#CC79A7'
  ]

  [size]
  figsize = [8.0, 4.0]      # [横幅, 縦幅] を表す数値
  fontsize = 16.0           # フォントサイズを表す数値
  linewidth = 1.0           # 線の太さを表す数値
  capsize = 1.0             # エラーバーのキャップの長さを表す数値

  [misc]
  dpi = 256.0               # 解像度を表す数値
  format = 'eps'            # 保存する際のフォーマットを表す文字列；pngなど