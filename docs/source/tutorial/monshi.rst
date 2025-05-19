質問紙の集計（monshi）
====================

このチュートリアルでは `こちらのCSVファイル <https://github.com/issakuss/issych/blob/master/tests/testdata/monshi/testdata.csv>`_ にある質問紙回答を集計します。
一列目には被験者のID、２列目にはタイムスタンプが入っています。
三列目以降には、Scale 1, Scale 2, Scale 3 という名前の質問紙の項目が入っています。
ただし、6列目にはダミー項目があり、これは今回の集計に使いません。
Monshi を使うことで `こちらのCSVファイル <https://github.com/issakuss/issych/blob/master/tests/testdata/monshi/testdata-manually-scored.csv>`_ にあるような集計結果を得ることができます。

回答の下処理をする
---------------

回答を集めた `.csv` ファイルを読み込み、Python で下処理をします。
Monshi では、回答がすべて数値または数値に変換可能な文字列である必要があります。
「1. Strongly disagree」といった回答は扱えないため、これを数字のみから成る文字列「1」に変換します。

.. code-block:: python

    answer_sheet = pd.read_csv('testdata.csv')  # 回答を読み込む
    answer_sheet = answer_sheet.astype(str)  # 型を文字列に変換する（下で.split()が使えるように）
    answer_sheet = answer_sheet.map(                      # answer_sheet の各要素に対して以下の処理をする
        lambda x: x if pd.isna(x) else x.split('.')[0]))  # 要素が NaN であればそのまま、そうでなければ . で分割して最初の要素を取る

monfig を作る
------------

Monshi 用の config、略して monfig を作ります。
monfig は、Monshi における集計方法情報を指します。
`.toml` ファイルまたは辞書型変数で指定することができます。
ここでは `.toml` ファイルを使いましょう。
`.toml` ファイルの例は以下の通りです。

.. code-block:: toml

    [_cols_item]
        # 回答を集めたデータフレームのうち、どの列までがどの質問紙に対応しているかを示す
        info = [0, 1]
        scale1 = ['C', 'D', 'E', 'G', 'H']
        scale2 = 'scale2_'
        scale3 = 'S:U'

    [info]
        idx_reverse = []

    [scale1]
        idx_reverse = [1, 2]
        min_plus_max = 6
        na_policy = 'ignore'
        average = true
        
    [scale2]
        preprocess = 'q > 3'
        idx_reverse = [1, 3, 4]
        min_plus_max = 1
        [scale2.subscale]
            a = [-1, 3, 5, 7, 9]
            b = [2, 4, 6, 8, 10]
            total = 'all'

    [scale3]
        idx_reverse = []

1. `[_cols_item]` で、列と質問紙の対応関係を示す

まず `[_cols_item]` というセクションを作ってください。
このセクションの名前は必ず `_cols_item` にしてください。
このセクションでは、（任意の）質問紙名と、その質問紙に対応する列を指定します。
列を指定する方法はいくつかあります。（ `参照 <https://issakuss.github.io/issych-docs/api/generated/issych.monshi.Monshi.separate.html>`_ ）

2. 各質問紙の集計方法を指定する

次に、以下の項目に入力することで、各質問紙の集計方法を指定してください。（ `参照 <https://issakuss.github.io/issych-docs/api/generated/issych.monshi.score_questionnaire.html>`_ ）

- `preprocess`: 前処理に用いる Python コードを指定します
- `idx_reverse`: 逆転項目のインデックスを指定します
- `min_plus_max`: 最小値と最大値を足した値を指定します。逆転項目がある場合には入力してください
- `na_policy`: 欠損値の扱いを指定します
- `subscale`: 下位尺度を指定します
- `average`: 合計値の代わりに平均値を計算する場合は `true` を指定します

3. Monshi を実行する

Monshi を実行するには、以下のようにします。（ `参照 <https://issakuss.github.io/issych-docs/api/generated/issych.monshi.score_as_monfig.html>`_ ）

.. code-block:: python

    from issych.monshi import score_as_monfig
    scores = score_as_monfig(answer_sheet, 'monfig.toml')

変数 `scores` に、集計結果が入ります。