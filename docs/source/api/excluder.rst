.. _api_excluder:

.. currentmodule:: issych.excluder

データ除外（excluder）
##################################################

データセットからデータを除外し、その過程を記録するためのモジュールです。

.. _excluder:

.. autosummary::
   :toctree: generated/

   DataExcluder
   DataExcluder.to_pickle
   DataExcluder.read_pickle
   DataExcluder.rm_byvalue
   DataExcluder.rm_na
   DataExcluder.rm_index
   DataExcluder.rm_index_except
   DataExcluder.rm_as
   DataExcluder.get_summary
   DataExcluder.get_ok_index
   DataExcluder.get_ng_index
   DataExcluder.get_df
   DataExcluder.retention_matrix
   DataExcluder.plot_summaries