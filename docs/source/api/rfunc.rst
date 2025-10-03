.. _api_rfunc:

.. currentmodule:: issych.rfunc

R を用いた解析（rfunc）
####################

R を使った解析を Python から行うためのモジュールです。
:py:class:`rpy2` を利用して R にアクセスします。
使用のためには、R といくつかの R パッケージが必要です。

.. note::
   :py:class:`issych` から直接インポートできません。
   ``from issych.rfunc import GlmmTMB`` のようにインポートしてください。

.. _rfunc:

.. autosummary::
   :toctree: generated/

   rver
   GlmmTMB