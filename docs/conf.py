# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../veloxml'))
sys.path.insert(0, os.path.abspath('../veloxml/core'))  # c_veloxml_core のパスを追加

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'VeloxML'
copyright = '2025, Yuji Chinen'
author = 'Yuji Chinen'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
  "sphinx.ext.autodoc",
  'sphinx.ext.viewcode',      # ソースコードのリンク
  "sphinx.ext.napoleon",
  "sphinx.ext.autosummary",
  "myst_parser"]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# autodoc の詳細設定
autodoc_default_options = {
    'members': True,             # クラスや関数の docstring を出力
    'undoc-members': True,       # docstring がないメンバーも出力
    'private-members': True,     # _ で始まる関数・変数も出力
    'special-members': '__init__', # __init__ の docstring も出力
    'show-inheritance': True,    # 継承関係を表示
}

# 型ヒントを説明として表示
autodoc_typehints = 'description'

# クラスの docstring をクラスの説明の前後に挿入
autoclass_content = 'both'


autodoc_typehints = 'description'  # 型ヒントを説明に含める

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

