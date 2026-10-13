"""15 - MLlib clustering (KMeans) and recommender (ALS).

Run:  python tutorials/15_ml_clustering_als.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.helpers import banner, section  # noqa: E402
from common.session import get_spark, stop_spark  # noqa: E402


def main() -> None:
    banner("15 - MLlib Clustering & Recommender")
    spark = get_spark("15-clustering-als")
    import numpy as np
    import pandas as pd
    from pyspark.sql import functions as F
    from pyspark.ml.feature import VectorAssembler, StandardScaler
    from pyspark.ml.clustering import KMeans
    from pyspark.ml.evaluation import ClusteringEvaluator

    # ===================================================== KMeans
    section("KMeans: cluster customers by spend behaviour")
    rng = np.random.default_rng(11)
    # 3 latent segments
    seg = rng.integers(0, 3, 3000)
    freq = np.where(seg == 0, rng.normal(20, 4, 3000),
             np.where(seg == 1, rng.normal(6, 2, 3000), rng.normal(12, 3, 3000)))
    basket = np.where(seg == 0, rng.normal(30, 6, 3000),
               np.where(seg == 1, rng.normal(120, 20, 3000), rng.normal(60, 10, 3000)))
    recency = np.where(seg == 0, rng.normal(5, 2, 3000),
                np.where(seg == 1, rng.normal(40, 10, 3000), rng.normal(15, 5, 3000)))

    df = spark.createDataFrame(pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(3000)],
        "freq": freq.clip(1, None).round(1),
        "basket": basket.clip(1, None).round(1),
        "recency": recency.clip(0, None).round(1),
    }))

    assembler = VectorAssembler(inputCols=["freq", "basket", "recency"], outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features")
    feat = scaler.fit(assembler.transform(df)).transform(assembler.transform(df))

    evaluator = ClusteringEvaluator(featuresCol="features")
    best_k, best_score, best_model = None, -1, None
    for k in (2, 3, 4, 5):
        km = KMeans(featuresCol="features", k=k, seed=11, maxIter=30).fit(feat)
        pred = km.transform(feat)
        score = evaluator.evaluate(pred)          # silhouette
        print(f"k={k}  silhouette={score:.3f}  WSSSE={km.summary.trainingCost:,.0f}")
        if score > best_score:
            best_k, best_score, best_model = k, score, km

    print(f"\nbest k={best_k} (silhouette {best_score:.3f})")
    clustered = best_model.transform(feat)
    (clustered.groupBy("prediction")
              .agg(F.round(F.avg("freq"), 1).alias("avg_freq"),
                   F.round(F.avg("basket"), 1).alias("avg_basket"),
                   F.round(F.avg("recency"), 1).alias("avg_recency"),
                   F.count("*").alias("customers"))
              .orderBy("prediction").show())

    # ===================================================== ALS recommender
    section("ALS: movie recommender from implicit ratings")
    rng = np.random.default_rng(5)
    n_users, n_items = 500, 200
    rows = []
    # give each user a favourite genre so there is structure to learn
    for u in range(n_users):
        fav = rng.integers(0, 5)
        for _ in range(30):
            it = int(rng.integers(fav * 40, (fav + 1) * 40)) if rng.random() < 0.7 \
                else int(rng.integers(0, n_items))
            rows.append((u, it, float(rng.integers(1, 6))))
    ratings = spark.createDataFrame(pd.DataFrame(rows, columns=["user", "item", "rating"]))
    train, test = ratings.randomSplit([0.8, 0.2], seed=5)

    from pyspark.ml.recommendation import ALS
    from pyspark.ml.evaluation import RegressionEvaluator

    als = ALS(userCol="user", itemCol="item", ratingCol="rating",
              rank=12, regParam=0.1, maxIter=10, coldStartStrategy="drop", seed=5)
    model = als.fit(train)
    preds = model.transform(test)
    rmse = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    print(f"ALS test RMSE = {rmse.evaluate(preds):.3f}")

    section("Top-5 recommendations for users 0..2")
    recs = model.recommendForAllUsers(5)
    for r in recs.filter("user <= 2").collect():
        items = [x["item"] for x in r["recommendations"]]
        print(f"user {r['user']}: {items}")

    section("Item factors (latent embeddings) for similarity")
    factors = model.itemFactors
    print("learned item factors:", factors.count(), "items x", len(factors.first()["features"]), "dims")

    stop_spark(spark)


if __name__ == "__main__":
    main()
