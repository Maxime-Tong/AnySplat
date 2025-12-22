#!/usr/bin/env bash

# Configuration - adjust paths/suffixes if needed
VR_BASE="/data/xthuang/dataset/vrnerf"
VR_SUFFIX="images-jpeg-1k/10"
MIP_BASE="/data/xthuang/dataset/3dgs"
MIP_SUFFIX="images_2"   # adjust if Mip-NeRF360 uses a different subfolder

vr_scenes=(apartment kitchen raf_furnishedroom)
mip_scenes=(bonsai counter kitchen room)

sparse=(80)
dense=(100)

RESULTS_FILE="results.txt"

# Dry-run by default. Pass --run to actually execute commands.
DRY_RUN=1
if [[ "${1:-}" == "--run" ]]; then
    DRY_RUN=0
fi

run_and_parse() {
    local cmd="$1"
    local dataset="$2"
    local scene="$3"
    local v="$4"

    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[DRY] $cmd"
        echo "[DRY] Would append: ${dataset}, ${scene}, ${v}, <inference_ms>, <render_ms>"
        return 0
    fi

    echo "[RUN] $cmd"
    tmp=$(mktemp)
    # capture both stdout and stderr
    eval "$cmd" >"$tmp" 2>&1
    # parse elapsed times (ms)
    inference_ms=$(sed -n 's/.*\[Running inference\] Elapsed time: \([0-9.]\+\) ms.*/\1/p' "$tmp")
    render_ms=$(sed -n 's/.*\[Rendering video\] Elapsed time: \([0-9.]\+\) ms.*/\1/p' "$tmp")

    # fallback: try lines without brackets (in case formatting differs)
    if [[ -z "$inference_ms" ]]; then
        inference_ms=$(sed -n 's/.*Running inference.*Elapsed time: \([0-9.]\+\) ms.*/\1/p' "$tmp")
    fi
    if [[ -z "$render_ms" ]]; then
        render_ms=$(sed -n 's/.*Rendering video.*Elapsed time: \([0-9.]\+\) ms.*/\1/p' "$tmp")
    fi

    # default to empty string if still not found
    inference_ms=${inference_ms:-}
    render_ms=${render_ms:-}

    printf '%s, %s, %s, %s, %s\n' "$dataset" "$scene" "$v" "$inference_ms" "$render_ms" >>"$RESULTS_FILE"

    rm -f "$tmp"
}

# VR-NeRF: sparse then dense
for scene in "${vr_scenes[@]}"; do
    data_path="${VR_BASE}/${scene}/${VR_SUFFIX}"
    dataset_label="vrnerf"
    for v in "${sparse[@]}"; do
        cmd="python inference.py --data \"$data_path\" --v $v"
        run_and_parse "$cmd" "$dataset_label" "$scene" "$v"
    done
    for v in "${dense[@]}"; do
        cmd="python inference.py --data \"$data_path\" --v $v"
        run_and_parse "$cmd" "$dataset_label" "$scene" "$v"
    done
done

# Mip-NeRF360: sparse then dense
for scene in "${mip_scenes[@]}"; do
    data_path="${MIP_BASE}/${scene}/${MIP_SUFFIX}"
    dataset_label="mipnerf"
    for v in "${sparse[@]}"; do
        cmd="python inference.py --data \"$data_path\" --v $v"
        run_and_parse "$cmd" "$dataset_label" "$scene" "$v"
    done
    for v in "${dense[@]}"; do
        cmd="python inference.py --data \"$data_path\" --v $v"
        run_and_parse "$cmd" "$dataset_label" "$scene" "$v"
    done
done

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Dry run complete. Rerun with: $0 --run"
else
    echo "Results appended to $RESULTS_FILE"
fi