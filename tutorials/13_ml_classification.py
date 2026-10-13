"""13 - MLlib classification: pipeline, evaluation, class imbalance.

Run:  python tutorials/13_ml_classification.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.helpers import banner, section  # noqa: E402
from common.session import get_spark, stop_spark  # noqa: E402


def main() -> None:
    banner("13 - MLlib Classification")
    spark = get_spark("13-ml-classification")
    import numpy as np
    import pandas as pd

    from pyspark.sql import functions as F
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import LogisticRegression, GBTClassifier
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml.feature import VectorAssembler, StandardScaler

    # --------------------------------------------------- synthetic dataset
    section("Synthetic binary-classification dataset")
    rng = np.random.default_rng(42)
    n = 8000
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.normal(0, 1, n)
    logit = 1.5 * x1 - 1.0 * x2 + 0.5 * x3 - 0.3
    p = 1 / (1 + np.exp(-logit))
    y = (rng.random(n) < p).astype(int)
    pdf = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "label": y})
    df = spark.createDataFrame(pdf)
    print("rows:", df.count(), "| positives:", df.filter("label=1").count())

    train, test = df.randomSplit([0.8, 0.2], seed=42)

    # class imbalance weighting
    pos = train.filter("label=1").count()
    neg = train.filter("label=0").count()
    weighted = train.withColumn(
        "weight", F.when(F.col("label") == 1, F.lit(neg / max(pos, 1))).otherwise(1.0))

    # --------------------------------------------------- pipelines
    section("Train logistic regression and gradient-boosted trees")
    assembler = VectorAssembler(inputCols=["x1", "x2", "x3"], outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features")

    auc = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")
    ap = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderPR")

    models = {}
    for name, clf in [
        ("logreg", LogisticRegression(featuresCol="features", labelCol="label",
                                      weightCol="weight", maxIter=50)),
        ("gbt", GBTClassifier(featuresCol="features", labelCol="label",
                              weightCol="weight", maxIter=30, seed=42)),
    ]:
        model = Pipeline(stages=[assembler, scaler, clf]).fit(weighted)
        preds = model.transform(test)
        models[name] = (model, preds)
        print(f"{name:7s} AUC={auc.evaluate(preds):.4f}  PR-AUC={ap.evaluate(preds):.4f}")

    # --------------------------------------------------- evaluate
    best_name = max(models, key=lambda k: auc.evaluate(models[k][1]))
    model, preds = models[best_name]
    section(f"Confusion matrix & metrics for best model ({best_name})")

    from pyspark.ml.functions import vector_to_array
    cm = (preds.withColumn("dec", F.col("prediction"))
               .groupBy("label", "dec").count().orderBy("label", "dec"))
    cm.show()

    tp = preds.filter("label=1 AND prediction=1").count()
    tn = preds.filter("label=0 AND prediction=0").count()
    fp = preds.filter("label=0 AND prediction=1").count()
    fn = preds.filter("label=1 AND prediction=0").count()
    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    print(f"accuracy={acc:.3f}  precision={prec:.3f}  recall={rec:.3f}")

    section("Threshold tuning (PD scores)")
    scored = preds.withColumn("score", vector_to_array("probability")[1])
    for t in (0.3, 0.5, 0.7):
        flagged = scored.filter(f"score >= {t}").count()
        print(f"threshold {t}: {flagged} flagged of {scored.count()}")

    section("Cross-validation (small grid)")
    from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
    lr = LogisticRegression(featuresCol="features", labelCol="label", weightCol="weight")
    grid = ParamGridBuilder().addGrid(lr.regParam, [0.0, 0.1]).build()
    cv = CrossValidator(estimator=Pipeline(stages=[assembler, scaler, lr]),
                        estimatorParamMaps=grid, evaluator=auc, numFolds=3, seed=42)
    cv_model = cv.fit(weighted)
    print(f"best regParam AUC={auc.evaluate(cv_model.transform(test)):.4f}")

    stop_spark(spark)


if __name__ == "__main__":
    main()
