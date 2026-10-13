"""14 - MLlib regression: linear regression, feature engineering, metrics.

Run:  python tutorials/14_ml_regression.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.helpers import banner, section  # noqa: E402
from common.session import get_spark, stop_spark  # noqa: E402


def main() -> None:
    banner("14 - MLlib Regression")
    spark = get_spark("14-ml-regression")
    import numpy as np
    import pandas as pd

    from pyspark.ml import Pipeline
    from pyspark.ml.feature import VectorAssembler, StandardScaler
    from pyspark.ml.regression import LinearRegression, GBTRegressor
    from pyspark.ml.evaluation import RegressionEvaluator

    # ---------------------------------------------- synthetic house prices
    section("Synthetic regression dataset (predict price)")
    rng = np.random.default_rng(7)
    n = 6000
    sqft = rng.normal(1800, 600, n).clip(300, 6000)
    beds = rng.integers(1, 6, n)
    age = rng.integers(0, 60, n)
    dist = rng.exponential(5, n).clip(0.1, 40)          # km to centre
    noise = rng.normal(0, 20000, n)
    price = (50_000 + 120 * sqft + 12_000 * beds - 1_500 * age - 8_000 * dist + noise)

    df = spark.createDataFrame(pd.DataFrame({
        "sqft": sqft, "beds": beds, "age": age, "dist": dist, "price": price}))

    train, test = df.randomSplit([0.8, 0.2], seed=7)

    # ---------------------------------------------- linear regression
    section("Linear regression (interpretable coefficients)")
    feats = ["sqft", "beds", "age", "dist"]
    assembler = VectorAssembler(inputCols=feats, outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)
    lr = LinearRegression(featuresCol="features", labelCol="price")

    lr_model = Pipeline(stages=[assembler, scaler, lr]).fit(train)
    lr_pred = lr_model.transform(test)

    rmse = RegressionEvaluator(labelCol="price", metricName="rmse")
    r2 = RegressionEvaluator(labelCol="price", metricName="r2")
    mae = RegressionEvaluator(labelCol="price", metricName="mae")
    print(f"LR   RMSE={rmse.evaluate(lr_pred):,.0f}  MAE={mae.evaluate(lr_pred):,.0f}  R2={r2.evaluate(lr_pred):.4f}")

    lr_stage = lr_model.stages[-1]
    print("coefficients (standardized):", [round(c, 1) for c in lr_stage.coefficients])
    print("intercept:", round(lr_stage.intercept, 1))

    # ---------------------------------------------- feature engineering
    section("Feature engineering: add derived features")
    from pyspark.sql import functions as F
    train2 = (train
              .withColumn("rooms_per_sqft", F.col("beds") / F.col("sqft"))
              .withColumn("age_sqft", F.col("age") * F.col("sqft")))
    test2 = (test
             .withColumn("rooms_per_sqft", F.col("beds") / F.col("sqft"))
             .withColumn("age_sqft", F.col("age") * F.col("sqft")))

    feats2 = feats + ["rooms_per_sqft", "age_sqft"]
    gbt = GBTRegressor(featuresCol="features", labelCol="price", maxIter=40, seed=7)
    gbt_model = Pipeline(stages=[
        VectorAssembler(inputCols=feats2, outputCol="features_raw"),
        StandardScaler(inputCol="features_raw", outputCol="features"),
        gbt,
    ]).fit(train2)
    gbt_pred = gbt_model.transform(test2)
    print(f"GBT  RMSE={rmse.evaluate(gbt_pred):,.0f}  MAE={mae.evaluate(gbt_pred):,.0f}  R2={r2.evaluate(gbt_pred):.4f}")

    # ---------------------------------------------- predictions
    section("Sample predictions")
    gbt_pred.select("sqft", "beds", "age", "dist", "price",
                    F.round("prediction", 0).alias("predicted")).show(8)

    section("Residual check")
    resid = gbt_pred.withColumn("residual", F.col("price") - F.col("prediction"))
    resid.select(F.round(F.mean("residual"), 1).alias("mean_resid"),
                 F.round(F.stddev("residual"), 1).alias("std_resid")).show()

    stop_spark(spark)


if __name__ == "__main__":
    main()
