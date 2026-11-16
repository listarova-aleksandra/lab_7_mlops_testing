import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, Check

FEATURES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]

# Декларативная схема валидации Pandera
iris_schema = pa.DataFrameSchema(
    {
        "sepal length (cm)": Column(float, [Check.ge(0.0), Check.le(10.0)], nullable=False),
        "sepal width (cm)": Column(float, [Check.ge(0.0), Check.le(10.0)], nullable=False),
        "petal length (cm)": Column(float, [Check.ge(0.0), Check.le(10.0)], nullable=False),
        "petal width (cm)": Column(float, [Check.ge(0.0), Check.le(10.0)], nullable=False),
    },
    strict=True,
)


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Проверяет DataFrame по схеме и выбрасывает SchemaError при нарушениях."""
    return iris_schema.validate(df)


def validate_with_great_expectations(df: pd.DataFrame) -> bool:
    """Проверяет качественные правила (Expectations) с помощью Great Expectations."""
    import great_expectations as ge

    if hasattr(ge, "from_pandas"):
        # Legacy API (great_expectations < 1.0)
        ge_df = ge.from_pandas(df)
        results = [ge_df.expect_column_to_exist(col) for col in FEATURES]
        results += [ge_df.expect_column_values_to_not_be_null(col) for col in FEATURES]
        results += [
            ge_df.expect_column_values_to_be_between(col, min_value=0.0, max_value=10.0)
            for col in FEATURES
        ]
        return all(r.success for r in results)

    # Новый API (great_expectations >= 1.0)
    import great_expectations.expectations as gxe

    context = ge.get_context(mode="ephemeral")
    source = context.data_sources.add_pandas("iris_source")
    asset = source.add_dataframe_asset("iris_asset")
    batch_def = asset.add_batch_definition_whole_dataframe("iris_batch")
    batch = batch_def.get_batch(batch_parameters={"dataframe": df})

    expectations = []
    for col in FEATURES:
        expectations.append(gxe.ExpectColumnToExist(column=col))
        expectations.append(gxe.ExpectColumnValuesToNotBeNull(column=col))
        expectations.append(gxe.ExpectColumnValuesToBeBetween(column=col, min_value=0.0, max_value=10.0))

    return all(batch.validate(e).success for e in expectations)
