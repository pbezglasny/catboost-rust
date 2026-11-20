"""Regenerate the CBM fixtures and print the raw Python predictions used by the Rust tests."""

from pathlib import Path

import catboost as cb

assert cb.__version__ == "1.2.10"
directory = Path(__file__).parent
options = dict(iterations=2, depth=1, learning_rate=0.5, random_seed=0,
               random_strength=0, bootstrap_type="No", thread_count=1,
               allow_writing_files=False, verbose=False)


def save(name, model, pool):
    model.fit(pool)
    model.get_metadata()["model_guid"] = name
    model.get_metadata()["train_finish_time"] = "1970-01-01T00:00:00Z"
    model.save_model(directory / f"{name}.cbm")


rows = [[0.0], [1.0], [2.0], [3.0]]
for name, loss, targets in [
    ("scalar", "RMSE", [0, 0, 4, 4]),
    ("multi", "MultiRMSE", [[0, 4], [0, 4], [4, 0], [4, 0]]),
]:
    model = cb.CatBoostRegressor(loss_function=loss, **options)
    save(name, model, cb.Pool(rows, targets))
    print(name, model.predict([rows[0], rows[-1]]).tolist())

labels = [0, 1] * 32
for name, data, kind in [
    ("categorical", [["a"], ["b"]], "cat_features"),
    ("text", [["red apple"], ["blue sky"]], "text_features"),
    ("embedding", [[[0.0, 0.2]], [[1.0, 1.5]]], "embedding_features"),
]:
    pool = cb.Pool(data * 32, labels, **{kind: [0]})
    save(name, cb.CatBoostClassifier(**options), pool)
