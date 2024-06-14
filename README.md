# uspto_balance

[![License](https://img.shields.io/pypi/l/uspto_balance.svg?color=green)](https://github.com/yvsgrndjn/uspto_balance/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/uspto_balance.svg?color=green)](https://pypi.org/project/uspto_balance)
[![Python Version](https://img.shields.io/pypi/pyversions/uspto_balance.svg?color=green)](https://python.org)
[![CI](https://github.com/yvsgrndjn/uspto_balance/actions/workflows/ci.yml/badge.svg)](https://github.com/yvsgrndjn/uspto_balance/actions/workflows/ci.yml)

Module to create similar reactions to a given retrosynthetic reaction template. First, the reaction template is being applied to molecules containing the same substructure as the product side. Second, each reaction will have its reagent(s) predicted before being validated by a disconnection-aware forward validation model. A reaction is considered valid if the original target product is found back during the forward validation with a confidence score higher than 95%.

## Installing
### From GitHub
~~~git clone git@github.com:yvsgrndjn/USPTO_balance.git
cd uspto_balance
conda create uspto_balance
conda activate uspto_balance
pip install -e . ~~~

