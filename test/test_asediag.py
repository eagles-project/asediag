#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_asediag
----------------------------------

Tests for `asediag` module.
"""
import pytest
from pathlib import Path

def test_importing():
    from asediag.aerosol_diag_SEgrid import gather_data, get_map, get_all_tables, get_forcing_df
    
