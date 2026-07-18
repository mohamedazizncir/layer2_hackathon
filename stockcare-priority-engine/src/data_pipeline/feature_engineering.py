from pathlib import Path

import pandas as pd


COLD_CHAIN_DCI_PATTERN = (
    r"INSULIN|SOMATROPINE|ERYTHROPOIETINE|EPOETINE|INTERFERON|FOLLITROPINE|"
    r"GONADOTROPHINE|OCYTOCINE|CALCITONINE"
)
INJECTABLE_REVIEW_PATTERN = r"injectable|perfusion|pdre p\.prep\.injectable"


def normalize_dosage(dosage: pd.Series) -> pd.Series:
    """Extract the first numeric amount and unit into a comparable key."""
    extracted = dosage.astype("string").str.extract(
        r"(?i)(?P<number>\d+(?:[.,]\d+)?)\s*(?P<unit>[a-zµμ%]+)",
        expand=True,
    )
    number = extracted["number"].str.replace(",", ".", regex=False)
    unit = extracted["unit"].str.lower()
    return (number + unit).astype("string")


def enrich_catalogue(catalogue: pd.DataFrame) -> pd.DataFrame:
    """Add substitution, irreplaceability, and cold-chain features."""
    enriched = catalogue.copy()
    enriched["dosage_key"] = normalize_dosage(enriched["dosage"])

    dci_key = enriched["dci"].astype("string").str.strip().str.upper()
    valid_group = dci_key.notna() & dci_key.ne("") & enriched["dosage_key"].notna()
    substitutes = pd.Series(0, index=enriched.index, dtype="int64")
    substitutes.loc[valid_group] = (
        enriched.loc[valid_group]
        .assign(_dci_key=dci_key.loc[valid_group])
        .groupby(["_dci_key", "dosage_key"], dropna=False)["dci"]
        .transform("size")
        .sub(1)
        .astype("int64")
    )
    enriched["n_substitutes"] = substitutes
    enriched["irreplaceability"] = 1.0 / (1.0 + enriched["n_substitutes"])

    dci_matches = dci_key.str.contains(
        COLD_CHAIN_DCI_PATTERN, case=False, na=False, regex=True
    ) | dci_key.str.endswith("MAB", na=False)
    vaccine_form = enriched["forme"].astype("string").str.contains(
        "vaccin", case=False, na=False, regex=False
    )
    biosimilar = (
        enriched["type"]
        .astype("string")
        .str.casefold()
        .eq("biosimilaire")
        .fillna(False)
    )

    enriched["cold_chain_sensitive"] = biosimilar | dci_matches | vaccine_form
    injectable_form = enriched["forme"].astype("string").str.contains(
        INJECTABLE_REVIEW_PATTERN, case=False, na=False, regex=True
    )
    enriched["cold_chain_needs_review"] = (
        injectable_form & ~enriched["cold_chain_sensitive"]
    )
    enriched["made_in_tunisia"] = (
        enriched["pays"].astype("string").str.strip().str.lower().eq("tunisie")
    ).fillna(False)

    enriched["criticality"] = pd.NA
    enriched["usage_type"] = pd.NA
    return enriched


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    input_path = project_root / "data" / "processed" / "amm_clean.parquet"
    output_path = project_root / "data" / "processed" / "enriched_catalogue.parquet"

    catalogue = pd.read_parquet(input_path)
    enriched = enrich_catalogue(catalogue)
    enriched.to_parquet(output_path, index=False)

    print(f"cold_chain_sensitive: {int(enriched['cold_chain_sensitive'].sum())}")
    print(f"cold_chain_needs_review: {int(enriched['cold_chain_needs_review'].sum())}")


if __name__ == "__main__":
    main()
