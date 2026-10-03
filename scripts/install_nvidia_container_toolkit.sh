#!/usr/bin/env bash
set -euo pipefail

TOOLKIT_VERSION="1.20.1-1"

if [[ ${EUID} -eq 0 ]]; then
  SUDO=()
else
  SUDO=(sudo)
fi

"${SUDO[@]}" apt-get update
"${SUDO[@]}" apt-get install -y --no-install-recommends ca-certificates curl gnupg2
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  "${SUDO[@]}" gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' | \
  "${SUDO[@]}" tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
"${SUDO[@]}" apt-get update
"${SUDO[@]}" apt-get install -y \
  "nvidia-container-toolkit=${TOOLKIT_VERSION}" \
  "nvidia-container-toolkit-base=${TOOLKIT_VERSION}" \
  "libnvidia-container-tools=${TOOLKIT_VERSION}" \
  "libnvidia-container1=${TOOLKIT_VERSION}"
"${SUDO[@]}" nvidia-ctk runtime configure --runtime=docker
"${SUDO[@]}" systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
