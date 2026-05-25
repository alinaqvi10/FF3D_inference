#!/usr/bin/env bash
# Jetson variant of run_docker_locally.sh
# Differences vs. the x86 script:
#   - Builds from Dockerfile.jetson (dustynv/pytorch:2.1-r36.2.0 base)
#   - Uses --runtime nvidia instead of --gpus all
#   - shm-size scaled to the 16 GB unified memory budget
set -euo pipefail

############################################
# Config (overridable via env)
############################################
IMAGE_NAME="${IMAGE_NAME:-forestformer-jetson}"
CONTAINER_NAME="${CONTAINER_NAME:-forestformer-jetson-c}"
DOCKERFILE="${DOCKERFILE:-Dockerfile.jetson}"

HOST_PROJECT_DIR="${HOST_PROJECT_DIR:-$(pwd)}"
HOST_BUCKET_IN="${HOST_BUCKET_IN:-/home/forecr/repos/TreeSegmentation/FF3D_inference/FF3D_oracle/bucket_in_folder}"
HOST_BUCKET_OUT="${HOST_BUCKET_OUT:-/home/forecr/repos/TreeSegmentation/FF3D_inference/FF3D_oracle/bucket_out_folder}"

CONTAINER_TEST_DATA="/workspace/data/ForAINetV2/test_data"
CONTAINER_OUTPUT_DIR="/workspace/work_dirs/output"

# Orin NX 16 GB unified — keep shm modest, leave RAM for the model
SHM_SIZE="${SHM_SIZE:-4g}"

REBUILD_IMAGE="${REBUILD_IMAGE:-auto}"      # auto|always|never
RECREATE_CONTAINER="${RECREATE_CONTAINER:-auto}"

SKIP_PROVISION="${SKIP_PROVISION:-false}"
SKIP_PREPROCESS="${SKIP_PREPROCESS:-false}"

printf "Current directory: %s\n" "$(pwd)"

############################################
# 0) Pre-check
############################################
docker info >/dev/null 2>&1 || { echo "Docker unavailable"; exit 1; }

# Confirm nvidia runtime present (Jetson default)
if ! docker info 2>/dev/null | grep -q 'Runtimes:.*nvidia'; then
  echo "WARNING: docker 'nvidia' runtime not detected. Install nvidia-container-toolkit."
fi

mkdir -p "${HOST_BUCKET_IN}" "${HOST_BUCKET_OUT}"

############################################
# 1) Build (on demand)
############################################
echo "[1/5] Build image: ${IMAGE_NAME} (from ${DOCKERFILE})"
image_exists="$(docker images -q "${IMAGE_NAME}" 2>/dev/null || true)"

should_build=false
case "$REBUILD_IMAGE" in
  always) should_build=true ;;
  never)  should_build=false ;;
  auto)   [[ -z "$image_exists" ]] && should_build=true || should_build=false ;;
  *) echo "Unknown REBUILD_IMAGE=${REBUILD_IMAGE} (use auto|always|never)"; exit 1 ;;
esac

if $should_build; then
  docker build -f "${DOCKERFILE}" -t "${IMAGE_NAME}" .
else
  echo "Image exists, skip build. (REBUILD_IMAGE=${REBUILD_IMAGE})"
fi

############################################
# 2) Ensure container + mount buckets
############################################
echo "[2/5] Ensure container: ${CONTAINER_NAME}"

run_container () {
  docker run -d \
    --runtime nvidia \
    --shm-size="${SHM_SIZE}" \
    --name "${CONTAINER_NAME}" \
    -v "${HOST_PROJECT_DIR}:/workspace" \
    --mount "type=bind,source=${HOST_BUCKET_IN},target=${CONTAINER_TEST_DATA}" \
    --mount "type=bind,source=${HOST_BUCKET_OUT},target=${CONTAINER_OUTPUT_DIR}" \
    --entrypoint bash \
    "${IMAGE_NAME}" -lc "sleep infinity"
}

container_exists="$(docker ps -a --format '{{.Names}}' | grep -x "${CONTAINER_NAME}" || true)"
container_running="$(docker ps --format '{{.Names}}' | grep -x "${CONTAINER_NAME}" || true)"

case "$RECREATE_CONTAINER" in
  always)
    [[ -n "$container_exists" ]] && docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    run_container
    ;;
  never)
    if [[ -z "$container_exists" ]]; then
      echo "Container not found and RECREATE_CONTAINER=never → creating once."
      run_container
    else
      [[ -z "$container_running" ]] && docker start "${CONTAINER_NAME}" >/dev/null
    fi
    ;;
  auto)
    if [[ -z "$container_exists" ]]; then
      run_container
    else
      [[ -z "$container_running" ]] && docker start "${CONTAINER_NAME}" >/dev/null
    fi
    ;;
  *) echo "Unknown RECREATE_CONTAINER=${RECREATE_CONTAINER} (use auto|always|never)"; exit 1 ;;
esac

echo "[2/5] Container is up (sleep infinity)."

############################################
# 3) Run the project entrypoint (handles patches + pipeline)
############################################
echo "[3/5] Run entrypoint in container"
docker exec -i \
  -e SKIP_PROVISION="${SKIP_PROVISION}" \
  -e SKIP_PREPROCESS="${SKIP_PREPROCESS}" \
  "${CONTAINER_NAME}" bash -lc "bash /workspace/entrypoint_ff3d.sh"

echo
echo "✅ Pipeline finished."
echo "Logs:  docker logs -f ${CONTAINER_NAME}"
echo "Shell: docker exec -it ${CONTAINER_NAME} /bin/bash"
