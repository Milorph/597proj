# run from repo root:  python -m supervised.train

import os
import joblib
from sklearn.model_selection import train_test_split

from src import data_loader, sampling, flow_aggregation, evaluate
from supervised.model import Prep, Classifier


def main():
    _, flows, src = data_loader.load_populations()      # reads real csvs from data/flow
    unified = flow_aggregation.aggregate_flows(flows)   # one record per flow
    df = sampling.sample_dataset(unified)               # 200k benign + 2-3% attacks

    tr, tmp = train_test_split(df, test_size=0.3, random_state=42,
                               stratify=df["attack_type"])
    val, te = train_test_split(tmp, test_size=0.5, random_state=42,
                               stratify=tmp["attack_type"])

    prep = Prep()
    Xtr = prep.fit_transform(tr)
    Xval = prep.transform(val)
    Xte = prep.transform(te)
    ytr, yval, yte = tr["Label"].to_numpy(), val["Label"].to_numpy(), te["Label"].to_numpy()

    clf = Classifier().fit(Xtr, ytr)
    thr = clf.tune(Xval, yval)
    m = evaluate.binary_metrics(yte, clf.predict(Xte, thr), clf.proba(Xte))
    print("F1=%.3f AUC=%.3f P=%.3f R=%.3f thr=%.2f"
          % (m["f1"], m["auc_roc"], m["precision"], m["recall"], thr))

    # save for the cascade later (preprocessor + classifier + threshold)
    os.makedirs("artifacts", exist_ok=True)
    joblib.dump((prep, clf, thr), "artifacts/supervised.joblib")
    print("saved artifacts/supervised.joblib")


if __name__ == "__main__":
    main()
