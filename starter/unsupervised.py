# phase 2 -- run:  python unsupervised.py   (from inside this folder)

import os
import joblib
from sklearn.model_selection import train_test_split

import common
from models import Prep, Detector, DROP_PKT


def main():
    print("loading packet data...")
    packets = common.load_packets()                   # data/packet/<attack>/*.csv
    df = common.sample_dataset(packets)               # 200k benign + 2-3% attacks
    tr, te = train_test_split(df, test_size=0.3, random_state=42,
                              stratify=df["attack_type"])

    prep = Prep(DROP_PKT)
    Xtr = prep.fit_transform(tr)
    Xte = prep.transform(te)

    det = Detector().fit(Xtr)                          # no labels passed
    pred = det.predict(Xte)
    scores = det.score(Xte)

    y = te["Label"].to_numpy()
    m = common.binary_metrics(y, pred, scores)
    print("P=%.3f R=%.3f F1=%.3f AUC=%.3f FP=%d"
          % (m["precision"], m["recall"], m["f1"], m["auc_roc"], m["fp"]))
    rates = common.per_attack_detection_rate(te["attack_type"].to_numpy(), pred)
    print({k: round(v["detection_rate"], 3) for k, v in rates.items()})

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump((prep, det), "artifacts/unsupervised.joblib")   # for the cascade later
    print("saved artifacts/unsupervised.joblib")


if __name__ == "__main__":
    main()
