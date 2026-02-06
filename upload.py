#!/usr/bin/env python3
"""
Upload Titanic Survival Model to ModelScope
Requires: modelscope library and logged-in CLI
Install: pip install modelscope
Login:   modelscope login
"""

import os
import shutil

MODEL_DIR = "models"
UPLOAD_DIR = "modelscope_upload"


def prepare_upload():
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    files_to_upload = [
        "inference.py",
        "requirements.txt",
        "README.md",
        "modelscope_config.json",
        "lifeboat_sdk/__init__.py",
        "lifeboat_sdk/lifeboat.py",
        "lifeboat_sdk/config.json",
        "lifeboat_sdk/setup.py",
        "lifeboat_sdk/requirements.txt",
    ]

    for f in files_to_upload:
        shutil.copy(f, os.path.join(UPLOAD_DIR, f))

    for f in os.listdir(MODEL_DIR):
        src = os.path.join(MODEL_DIR, f)
        dst = os.path.join(UPLOAD_DIR, f)
        shutil.copy(src, dst)

    sdk_pkl_files = [
        "features.pkl",
        "scaler.pkl",
        "le_sex.pkl",
        "le_embarked.pkl",
        "titanic_model.pkl",
    ]
    for f in sdk_pkl_files:
        src = os.path.join("lifeboat_sdk", f)
        dst = os.path.join(UPLOAD_DIR, f)
        if os.path.exists(src):
            shutil.copy(src, dst)

    with open(os.path.join(UPLOAD_DIR, ".gitignore"), "w") as f:
        f.write("__pycache__/\n*.pyc\n.venv/\n")

    print(f"Prepared files in: {UPLOAD_DIR}/")
    print("Ready for ModelScope upload!")


def upload_to_modelscope(repo_id="bniladridas/lifeboat"):
    from modelscope.hub.api import HubApi

    prepare_upload()

    api = HubApi()
    api.upload_folder(
        folder_path=UPLOAD_DIR,
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload Titanic Survival Prediction Model",
    )
    print(f"Uploaded to: https://modelscope.cn/models/{repo_id}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        repo_id = sys.argv[1]
        upload_to_modelscope(repo_id)
    else:
        prepare_upload()
        print("\nRun: python upload.py <your-username>/titanic-survival-prediction")
