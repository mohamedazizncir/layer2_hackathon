from pathlib import Path

import pandas as pd


EXPECTED_COLUMNS = [
    "nom_specialite",
    "dosage",
    "forme",
    "presentation",
    "dci",
    "laboratoire",
    "pays",
    "num_amm",
    "date_amm",
    "type",
]

TYPE_NORMALIZATION = {
    "G/P": "Generique",
    "G": "Generique",
    "P": "Princeps",
}


def load_amm_registry(xlsx_path, sheet_name) -> pd.DataFrame:
    """Load and clean a sheet from the DPM Tunisia AMM registry."""
    catalogue = pd.read_excel(xlsx_path, sheet_name=sheet_name)

    if len(catalogue.columns) != len(EXPECTED_COLUMNS):
        raise ValueError(
            f"Expected {len(EXPECTED_COLUMNS)} AMM columns in the documented order, "
            f"got {len(catalogue.columns)}: {list(catalogue.columns)}"
        )
    # The published workbook uses French display labels; its documented column
    # order is the stable schema contract used by the pipeline.
    catalogue.columns = EXPECTED_COLUMNS

    string_columns = catalogue.select_dtypes(include=["object", "string"]).columns
    for column in string_columns:
        catalogue[column] = catalogue[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )

    # Drop rows whose cells are all missing or blank after whitespace stripping.
    fully_empty = (catalogue.isna() | catalogue.eq("")).all(axis=1)
    catalogue = catalogue.loc[~fully_empty].reset_index(drop=True)

    catalogue["type"] = catalogue["type"].replace(TYPE_NORMALIZATION)
    catalogue["date_amm"] = pd.to_datetime(
        catalogue["date_amm"], errors="coerce", format="mixed", dayfirst=True
    )

    return catalogue


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    input_path = project_root / "data" / "raw" / "Medicaments_Tunisie_AMM.xlsx"
    output_path = project_root / "data" / "processed" / "amm_clean.parquet"

    catalogue = load_amm_registry(input_path, sheet_name=0)
    assert len(catalogue) == 6058, f"Expected 6058 rows, got {len(catalogue)}"
    assert catalogue["dci"].nunique() == 1088, (
        f"Expected 1088 unique DCI values, got {catalogue['dci'].nunique()}"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    catalogue.to_parquet(output_path, index=False)


if __name__ == "__main__":
    main()
