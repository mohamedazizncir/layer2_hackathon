from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


ANNOTATION_COLUMNS = {"dci", "atc_class", "usage_type", "criticality_score"}


def _normalize_dci(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip().str.upper()


def aggregate_dci_properties(catalogue: pd.DataFrame, include_criticality=False):
    """Aggregate shared catalogue properties to one row per normalized DCI."""
    work = catalogue.copy()
    work["_dci_key"] = _normalize_dci(work["dci"])
    type_key = work["type"].astype("string").str.strip().str.casefold()
    work["_is_princeps"] = type_key.eq("princeps").fillna(False).astype(float)
    work["_is_biosimilaire"] = (
        type_key.eq("biosimilaire").fillna(False).astype(float)
    )
    aggregations = {
        "irreplaceability": ("irreplaceability", "mean"),
        "n_substitutes": ("n_substitutes", "min"),
        "cold_chain_sensitive": ("cold_chain_sensitive", "max"),
        "share_princeps": ("_is_princeps", "mean"),
        "share_biosimilaire": ("_is_biosimilaire", "mean"),
        "made_in_tunisia": ("made_in_tunisia", "mean"),
        "n_amm": ("_dci_key", "size"),
    }
    if include_criticality:
        aggregations["criticality"] = ("criticality", "first")
    return work.groupby("_dci_key", as_index=False, dropna=False).agg(**aggregations)


def build_dci_features(catalogue: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the AMM catalogue into one numeric feature row per DCI."""
    work = catalogue.copy()
    work["_dci_key"] = _normalize_dci(work["dci"])

    forme_key = work["forme"].astype("string").str.strip().str.upper()
    work["_forme_key"] = forme_key.fillna("AUTRE").replace("", "AUTRE")
    top_formes = set(work["_forme_key"].value_counts().head(15).index)

    # pandas mode is sorted, which gives deterministic tie-breaking.
    modal_forme = work.groupby("_dci_key", dropna=False)["_forme_key"].agg(
        lambda values: values.mode().iloc[0]
    )
    modal_forme = modal_forme.where(modal_forme.isin(top_formes), "AUTRE")

    features = aggregate_dci_properties(catalogue).set_index("_dci_key")
    features["most_common_forme"] = modal_forme
    features = features.reset_index()

    forme_dummies = pd.get_dummies(
        features.pop("most_common_forme"), prefix="forme", dtype=float
    )
    features = pd.concat([features, forme_dummies], axis=1)
    features["cold_chain_sensitive"] = features[
        "cold_chain_sensitive"
    ].astype(float)
    features["made_in_tunisia"] = features["made_in_tunisia"].astype(float)
    return features


def load_annotations(path: Path) -> pd.DataFrame:
    annotations = pd.read_csv(path)
    missing = ANNOTATION_COLUMNS.difference(annotations.columns)
    if missing:
        raise ValueError(f"Annotation file is missing columns: {sorted(missing)}")

    annotations = annotations.copy()
    annotations["_dci_key"] = _normalize_dci(annotations["dci"])
    if annotations["_dci_key"].duplicated().any():
        duplicates = annotations.loc[
            annotations["_dci_key"].duplicated(keep=False), "dci"
        ].tolist()
        raise ValueError(f"Annotations contain duplicate DCI values: {duplicates}")

    annotations["criticality_score"] = pd.to_numeric(
        annotations["criticality_score"], errors="raise"
    )
    if not annotations["criticality_score"].between(0, 1).all():
        raise ValueError("criticality_score values must all be between 0 and 1")
    return annotations


def train_and_score(
    catalogue: pd.DataFrame, annotations: pd.DataFrame
) -> tuple[pd.DataFrame, lgb.LGBMRegressor]:
    features = build_dci_features(catalogue)
    feature_columns = [column for column in features if column != "_dci_key"]

    ground_truth = annotations[["_dci_key", "criticality_score"]]
    training = features.merge(ground_truth, on="_dci_key", how="inner")
    if training.empty:
        raise ValueError("No annotation DCI values overlap the AMM catalogue")

    annotated_not_found = set(ground_truth["_dci_key"]) - set(features["_dci_key"])
    if annotated_not_found:
        print(
            "Warning: annotations absent from catalogue: "
            f"{len(annotated_not_found)}"
        )

    x_train, x_test, y_train, y_test = train_test_split(
        training[feature_columns],
        training["criticality_score"],
        test_size=0.2,
        random_state=42,
    )
    model = lgb.LGBMRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        min_child_samples=5,
        random_state=42,
        verbosity=-1,
    )
    model.fit(x_train, y_train)

    test_predictions = model.predict(x_test)
    print(f"Test MAE: {mean_absolute_error(y_test, test_predictions):.6f}")
    print(f"Test R2: {r2_score(y_test, test_predictions):.6f}")

    importances = (
        pd.Series(model.feature_importances_, index=feature_columns)
        .sort_values(ascending=False)
    )
    print("Feature importances:")
    for feature, importance in importances.items():
        print(f"  {feature}: {int(importance)}")

    dci_scores = features[["_dci_key", "made_in_tunisia"]].copy()
    dci_scores["_prediction"] = model.predict(features[feature_columns])
    dci_scores = dci_scores.merge(ground_truth, on="_dci_key", how="left")
    dci_scores["criticality_is_predicted"] = dci_scores[
        "criticality_score"
    ].isna()
    dci_scores["criticality"] = dci_scores["criticality_score"].fillna(
        dci_scores["_prediction"]
    )

    base = catalogue.drop(
        columns=["criticality", "usage_type", "made_in_tunisia"], errors="ignore"
    ).copy()
    base["_dci_key"] = _normalize_dci(base["dci"])
    scored = base.merge(
        dci_scores[
            [
                "_dci_key",
                "criticality",
                "criticality_is_predicted",
                "made_in_tunisia",
            ]
        ],
        on="_dci_key",
        how="left",
        validate="many_to_one",
        sort=False,
    ).drop(columns="_dci_key")

    ground_truth_keys = set(ground_truth["_dci_key"])
    scored_keys = _normalize_dci(scored["dci"])
    assert not scored.loc[
        scored_keys.isin(ground_truth_keys), "criticality_is_predicted"
    ].any(), "A ground-truth DCI was incorrectly marked as predicted"
    assert scored["criticality"].notna().all(), "Some catalogue rows were not scored"

    return scored, model


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    processed_dir = project_root / "data" / "processed"
    catalogue_path = processed_dir / "enriched_catalogue.parquet"
    annotations_path = processed_dir / "criticality_annotations.csv"
    output_path = processed_dir / "catalogue_scored.parquet"

    catalogue = pd.read_parquet(catalogue_path)
    annotations = load_annotations(annotations_path)
    scored, _ = train_and_score(catalogue, annotations)
    scored.to_parquet(output_path, index=False)


if __name__ == "__main__":
    main()
