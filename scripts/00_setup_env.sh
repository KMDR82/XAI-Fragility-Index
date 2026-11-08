#!/usr/bin/env bash
# Phase 0 / step 1: create the pinned conda environment "xfi" (D-18).
# Run from the repository root inside WSL2 Ubuntu:   bash scripts/00_setup_env.sh
set -euo pipefail

ENV_NAME=xfi
eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[info] conda env '$ENV_NAME' already exists, reusing it"
else
  conda create -y -n "$ENV_NAME" python=3.12
fi
conda activate "$ENV_NAME"

python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-r1.txt

python -m ipykernel install --user --name "$ENV_NAME" --display-name "Python (xfi)"

python -m pip freeze > requirements-lock.txt
sha256sum requirements-lock.txt

python - <<'EOF'
import torch, torchvision, numpy, scipy, skimage, sklearn, cv2, pydicom, pytorch_grad_cam
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
print("torchvision", torchvision.__version__, "| numpy", numpy.__version__, "| scipy", scipy.__version__)
print("skimage", skimage.__version__, "| sklearn", sklearn.__version__, "| cv2", cv2.__version__, "| pydicom", pydicom.__version__)
EOF
echo "[done] environment ready. Select kernel 'Python (xfi)' in VS Code."
