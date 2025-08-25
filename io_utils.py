"""Utility functions for loading simulation inputs from spreadsheets."""
from __future__ import annotations

from typing import Dict

import pandas as pd

# Mapping of column headers to species names for Stage 1 and Stage 2
STAGE1_COLUMN_MAP = {
    "MnO2": "MnO2",
    "Mn2O3": "Mn2O3",
    "Mn3O4": "Mn3O4",
    "MnO": "MnO",
    "H2": "H2",
    "H2O": "H2O",
}

STAGE2_COLUMN_MAP = {
    "Al": "Al",
    "Fe": "Fe",
    "Si": "Si",
    "Mn": "Mn",
    "C": "C",
}


def _extract_stage_data(df: pd.DataFrame, column_map: Dict[str, str]) -> Dict[str, float]:
    """Convert a dataframe row to a composition dictionary.

    Parameters
    ----------
    df:
        DataFrame containing a single row with species masses.
    column_map:
        Mapping from dataframe column headers to species names used by the
        simulator.

    Returns
    -------
    Dict[str, float]
        Dictionary of species masses filtered and renamed according to the
        provided column mapping.
    """
    if df.empty:
        return {}
    row = df.iloc[0].to_dict()
    return {species: float(row[col]) for col, species in column_map.items() if col in row}


def load_stage_inputs(path: str) -> Dict[str, Dict[str, float]]:
    """Load Stage 1 and Stage 2 input compositions from an Excel file.

    The workbook is expected to contain separate sheets for each stage. Sheets
    whose name includes ``"stage1"`` (case-insensitive) are interpreted as
    Stage 1 data, and sheets including ``"stage2"`` are used for Stage 2. If
    such sheet names are not found, the first sheet is used for Stage 1 and the
    second for Stage 2.

    Parameters
    ----------
    path:
        Path to the Excel workbook.

    Returns
    -------
    Dict[str, Dict[str, float]]
        A dictionary with two keys, ``"stage1"`` and ``"stage2"``, each
        containing a dictionary of species masses compatible with
        :func:`run_stage1_simulation` and :func:`run_stage2_simulation`.
    """
    sheets = pd.read_excel(path, sheet_name=None)

    stage1_df = None
    stage2_df = None
    for name, df in sheets.items():
        lname = name.lower()
        if stage1_df is None and "stage1" in lname:
            stage1_df = df
        elif stage2_df is None and "stage2" in lname:
            stage2_df = df

    # Fallback: use first and second sheets
    if stage1_df is None:
        stage1_df = next(iter(sheets.values()))
    if stage2_df is None:
        remaining = [df for df in sheets.values() if df is not stage1_df]
        stage2_df = remaining[0] if remaining else stage1_df

    stage1_inputs = _extract_stage_data(stage1_df, STAGE1_COLUMN_MAP)
    stage2_inputs = _extract_stage_data(stage2_df, STAGE2_COLUMN_MAP)

    return {"stage1": stage1_inputs, "stage2": stage2_inputs}
