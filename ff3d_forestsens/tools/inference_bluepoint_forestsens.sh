#!/usr/bin/env bash
# inference_bluepoint.sh
# Description:
# 1) Place all input .ply files into /workspace/data/ForAINetV2/test_data/
# 2) This script automatically generates test_list.txt (overwriting any previous version),
#    copies it as test_list_initial.txt, and then runs the inference pipeline.
# 3) Final results are stored in BLUEPOINTS_DIR.

set -euo pipefail

########## Configurable Parameters (can be overridden by environment variables) ##########
WORK_DIR="${WORK_DIR:-/workspace}"
TEST_DATA_DIR="${TEST_DATA_DIR:-$WORK_DIR/data/ForAINetV2/test_data}"
TEST_DATA_DIR2="${TEST_DATA_DIR2:-$WORK_DIR/data/ForAINetV2/test_data}"
META_DIR="${META_DIR:-$WORK_DIR/data/ForAINetV2/meta_data}"

CONFIG_FILE="${CONFIG_FILE:-$WORK_DIR/configs/oneformer3d_qs_radius16_qp300_2many.py}"
MODEL_PATH="${MODEL_PATH:-$WORK_DIR/work_dirs/clean_forestformer/epoch_3000_fix.pth}"

# Number of inference iterations per scan
ITERATIONS="${ITERATIONS:-1}"

# Output directory for all inference results
BLUEPOINTS_DIR="${BLUEPOINTS_DIR:-$WORK_DIR/work_dirs/output}"

# GPU ID to use (default: 1)
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Reduce allocator fragmentation: GC at 80% capacity, 512 MB max split.
# Has no effect on model outputs — purely a memory management hint.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-garbage_collection_threshold:0.8,max_split_size_mb:512}"

# Internal file paths
TEST_LIST="$META_DIR/test_list.txt"
TEST_LIST_INIT="$META_DIR/test_list_initial.txt"

########## Prepare environment ##########
mkdir -p "$META_DIR" "$TEST_DATA_DIR" "$BLUEPOINTS_DIR" "$TEST_DATA_DIR2"

echo "[Step 0] Cleaning temporary data files in /app/code_binbin/ff3d_forestsens/data/ForAINetV2/forainetv2_instance_data ..."
rm -rf "$WORK_DIR/data/ForAINetV2/forainetv2_instance_data"/* || true
rm -rf "$WORK_DIR/data/ForAINetV2/semantic_mask"/* || true
rm -rf "$WORK_DIR/data/ForAINetV2/points"/* || true
rm -rf "$WORK_DIR/data/ForAINetV2/instance_mask"/* || true

########## Step 1: Generate test list (overwrite mode) ##########
# This overwrites test_list.txt completely each run (no appending).
echo "[Step 1] Generating test_list.txt (sorted by file size, overwrite mode)..."
python "$WORK_DIR/data/ForAINetV2/create_testlist.py"
cp -f "$TEST_LIST" "$TEST_LIST_INIT"

########## Step 1.5: Inline Safe-Name Sanitizer (no external script) ##########
# Rules:
#  • Replace "_bluepoints_" → "_bp_"
#  • If the basename ends with _<digits> and does not already end with "fixedname", append "fixedname"
#  • Collapse repeated "fixedname" → single "fixedname"
#  • Update test_list_initial.txt while preserving order
#  • Rename all files/directories in TEST_DATA_DIR accordingly (depth-first)
FIXTAG="${FIXTAG:-fixedname}"
LIST="$TEST_LIST_INIT"
LIST_BAK="${META_DIR}/test_list_initial_original.txt"
DATA="$TEST_DATA_DIR"

echo "[Step 1.5] Sanitizing names in list & ${DATA} (FIXTAG='${FIXTAG}')"
cp -f "$LIST" "$LIST_BAK" || true
echo "✔  Backup created → $(basename "$LIST_BAK")"

convert_base() {
  local base="$1"

  # 0) Replace "_bluepoints_" with "_bp_"
  local clean
  clean=$(echo "$base" | sed -E 's/_bluepoints_/_bp_/g')

  # 1) If it ends with _<digits> and does not already end with FIXTAG → append FIXTAG
  if [[ "$clean" =~ _[0-9]+$ ]] && [[ "$clean" != *"${FIXTAG}" ]]; then
    clean="${clean}${FIXTAG}"
  fi

  # 2) Collapse repeated FIXTAG occurrences (avoid multiple concatenations)
  clean=$(echo "$clean" | sed -E "s/(${FIXTAG})+/${FIXTAG}/g")

  echo "$clean"
}

tmp_list="${LIST}.tmp"
: > "$tmp_list"

# Declare associative array for "original_basename → safe_basename" mapping (requires bash 4+)
declare -A rename_map

# test_list_initial.txt typically contains basenames (without file extensions)
while IFS= read -r scan || [ -n "${scan:-}" ]; do
  safe_name="$(convert_base "$scan")"
  echo "$safe_name" >> "$tmp_list"
  rename_map["$scan"]="$safe_name"
done < "$LIST"

mv -f "$tmp_list" "$LIST"
echo "✔  $(basename "$LIST") updated ( $(wc -l < "$LIST") lines )"

# Traverse TEST_DATA_DIR (depth-first) and rename files/directories accordingly
find "$DATA" -depth | while read -r path; do
  name="$(basename "$path")"
  dir="$(dirname "$path")"

  base="$name"
  ext=""
  if [[ -f "$path" && "$name" == *.* ]]; then
    ext=".${name##*.}"
    base="${name%.*}"
  fi

  # Apply sanitization rules
  target_base="$(convert_base "$base")"
  # If a mapping exists from the test list, prefer that
  if [[ -n "${rename_map[$base]+x}" ]]; then
    target_base="${rename_map[$base]}"
  fi

  target="${target_base}${ext}"
  if [[ "$name" != "$target" ]]; then
    if [[ -e "$dir/$target" ]]; then
      echo "⚠  Skip (target already exists): $dir/$target"
    else
      mv "$path" "$dir/$target"
      echo "✔  mv $name → $target"
    fi
  fi
done
echo "[Step 1.5] Name sanitization completed."

########## Step 2: Inference loop ##########
echo "[Step 2] Running inference for each scan (ITERATIONS = $ITERATIONS)..."

# Iterate over all entries in test_list_initial.txt
while IFS= read -r scan_name || [ -n "$scan_name" ]; do
    echo "Processing: $scan_name"
    
    iteration=1
    current_scan_name="$scan_name"
    
    while [ "$iteration" -le "$ITERATIONS" ]; do
        echo "Iteration $iteration for $current_scan_name"
        
        # Update test_list.txt for this scan
        echo "$current_scan_name" > "$TEST_LIST"
        
        # Load test data
        cd "$WORK_DIR/data/ForAINetV2" || exit
        python batch_load_ForAINetV2_data.py --test_scan_names_file meta_data/test_list.txt
        cd "$WORK_DIR" || exit
        
        # Create intermediate data
        python "$WORK_DIR/tools/create_data_forainetv2.py" forainetv2

        # Adjust threshold (if needed)
        #score_th=$(echo "scale=2; 0.2 + ($iteration - 1) * 0.1" | bc)
        score_th=0.4

        # Update score_th in CONFIG_FILE
        sed -i "s/score_th = [0-9.]\+/score_th = 0$score_th/g" "$CONFIG_FILE"
        
        # Run model inference
        echo ">>> RUN: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python tools/test.py \"$CONFIG_FILE\" \"$MODEL_PATH\""
        CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python "$WORK_DIR/tools/test.py" "$CONFIG_FILE" "$MODEL_PATH"
        
        # New prediction file
        new_pre_FILE="${scan_name}_${iteration}.ply"
        new_pre_PATH="$BLUEPOINTS_DIR/$new_pre_FILE"
        
        if [ ! -f "$new_pre_PATH" ]; then
            echo "No prediction file found. Ending iterations for $scan_name."
            break
        fi
        
        # Bluepoints file
        BLUEPOINTS_FILE="${scan_name}_bluepoints_${iteration}.ply"
        BLUEPOINTS_PATH="$BLUEPOINTS_DIR/$BLUEPOINTS_FILE"
        
        # Verify existence of the bluepoints file
        if [ ! -f "$BLUEPOINTS_PATH" ]; then
            echo "No bluepoints file found. Ending iterations for $scan_name."
            break
        fi
        
        # Copy bluepoints file to test data directories for next iteration
        cp "$BLUEPOINTS_PATH" "$TEST_DATA_DIR/"
        cp "$BLUEPOINTS_PATH" "$TEST_DATA_DIR2/"
        
        # Prepare for the next round
        current_scan_name="${BLUEPOINTS_FILE%.ply}"
        ((iteration++))
    done

    # Merge predictions across iterations for the current scan
    echo "Merging results for $scan_name" 
    python tools/merge_prediction.py "$scan_name" "$BLUEPOINTS_DIR" "$ITERATIONS" 
    
    echo "Finished processing $scan_name. Moving to next."

done < "$TEST_LIST_INIT"

echo "All test cases processed successfully."
