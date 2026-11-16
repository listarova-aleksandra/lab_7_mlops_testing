import numpy as np
import pandas as pd
import pandera.pandas as pa
import pytest
from scipy.stats import ks_2samp
from sklearn.datasets import load_iris

from data_validation import validate_data, validate_with_great_expectations


@pytest.fixture
def iris_df() -> pd.DataFrame:
    return load_iris(as_frame=True).frame.drop(columns="target")


# --- 1. ТЕСТЫ ВАЛИДАЦИИ СХЕМЫ (PANDERA) ---
def test_valid_data_pandera(iris_df):
    validated_df = validate_data(iris_df)
    assert not validated_df.empty


def test_invalid_data_pandera():
    invalid_df = pd.DataFrame({
        "sepal length (cm)": [-5.0],  # Отрицательное значение!
        "sepal width (cm)": [3.0],
        "petal length (cm)": [1.4],
        "petal width (cm)": [0.2],
    })
    with pytest.raises(pa.errors.SchemaError):
        validate_data(invalid_df)


def test_missing_column_pandera(iris_df):
    with pytest.raises(pa.errors.SchemaError):
        validate_data(iris_df.drop(columns="petal width (cm)"))


# --- 2. ТЕСТЫ GREAT EXPECTATIONS ---
def test_great_expectations_checks(iris_df):
    assert validate_with_great_expectations(iris_df) is True


def test_great_expectations_detects_nulls(iris_df):
    broken = iris_df.copy()
    broken.loc[0, "petal length (cm)"] = np.nan
    assert validate_with_great_expectations(broken) is False


# --- 3. ТЕСТ НА ОБНАРУЖЕНИЕ ДРЕЙФА ДАННЫХ (KS-ТЕСТ) ---
def test_data_drift_detection(iris_df):
    reference = iris_df.sample(75, random_state=42)
    current_normal = iris_df.drop(reference.index)

    for col in reference.columns:
        p_value = ks_2samp(reference[col], current_normal[col]).pvalue
        assert p_value > 0.01, f"Неожиданный дрейф в {col}: p={p_value:.4f}"


def test_simulated_drift_failure(iris_df):
    reference = iris_df.sample(75, random_state=42)

    # Имитируем дрейф признака
    current_drifted = iris_df.drop(reference.index).copy()
    current_drifted["petal length (cm)"] *= 3.0

    p_value = ks_2samp(reference["petal length (cm)"], current_drifted["petal length (cm)"]).pvalue
    assert p_value < 0.05  # Дрейф успешно обнаружен
