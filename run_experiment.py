# 파일명: run_experiment.py
import subprocess
import os
import glob
import argparse
import shlex

# ==============================================================================
# ⚙️ 실험 설정
# ==============================================================================
EXPERIMENT_CONFIGS = {
    "blender_synthetic": {
        "base_scene_dir": "data/nerf_synthetic",
        "hdr_source_dirs": ["hdri"],
        "target_hdrs": "all",
        "output_dir": "output_blender",
        "train_flags": ["--random_background", "--hdr_rotation"]
    },
    "shiny_blender": {
        "base_scene_dir": "data/shiny_blender_dataset",
        "hdr_source_dirs": ["high_res_envmaps_2k"],
        "target_hdrs": ["city.hdr", "fireplace.hdr", "forest.hdr", "bridge.hdr"],
        "output_dir": "output_shiny_blender",
        "train_flags": ["--random_background", "--hdr_rotation"]
    }
}
# ==============================================================================

def run_command(command, is_dry_run=False):
    command_str = ' '.join(shlex.quote(str(arg)) for arg in command)
    if is_dry_run:
        print(f"🐫 [DRY RUN] Would execute: {command_str}")
        return
    print(f"\n▶️ Executing: {command_str}")
    try:
        subprocess.run(command, check=True, text=True)
        print("✅ Command finished successfully.")
    except Exception as e:
        print(f"❌ ERROR: {e}")
        raise

def main(args):
    config_name = args.config
    config = EXPERIMENT_CONFIGS[config_name]
    print(f"🚀 Running experiment with configuration: '{config_name}'")

    if args.dry_run:
        print("\n" + "#"*60 + "\n🐫 DRY RUN MODE ENABLED\n" + "#"*60 + "\n")
    
    os.makedirs(config["output_dir"], exist_ok=True)

    scenes_to_process = [d for d in os.listdir(config["base_scene_dir"]) if os.path.isdir(os.path.join(config["base_scene_dir"], d))]
    
    hdrs_to_process = []
    if config["target_hdrs"] == "all":
        for source_dir in config["hdr_source_dirs"]:
            hdrs_to_process.extend(glob.glob(os.path.join(source_dir, '*.hdr')))
    else:
        for source_dir in config["hdr_source_dirs"]:
            for hdr_name in config["target_hdrs"]:
                hdr_path = os.path.join(source_dir, hdr_name)
                if os.path.exists(hdr_path):
                    hdrs_to_process.append(hdr_path)

    print(f"Found {len(scenes_to_process)} scenes and {len(hdrs_to_process)} HDRs to process.")

    for scene_name in scenes_to_process:
        print("\n" + "="*60 + f"\nProcessing Scene: [{scene_name}]\n" + "="*60)
        
        scene_path = os.path.join(config["base_scene_dir"], scene_name)
        model_path = os.path.join(config["output_dir"], f"{scene_name}_model")
        
        # 1. 모델 훈련
        train_cmd = ["python", "train.py", "-s", scene_path, "-m", model_path, "--eval"]
        train_cmd.extend(config["train_flags"])
        run_command(train_cmd, is_dry_run=args.dry_run)
        
        # 2. 원본(GT) 이미지 렌더링
        gt_render_cmd = ["python", "render.py", "-m", model_path, "--skip_train", "--save_name", "ground_truth"]
        run_command(gt_render_cmd, is_dry_run=args.dry_run)
        
        # 3. 재조명 렌더링
        for hdr_path in hdrs_to_process:
            hdr_basename = os.path.splitext(os.path.basename(hdr_path))[0]
            relight_save_name = f"{scene_name}_{hdr_basename}"
            relight_cmd = ["python", "render.py", "-m", model_path, "--skip_train", "--save_name", relight_save_name, "-w", "--hdr_rotation", "--environment_texture", hdr_path, "--render_relight"]
            run_command(relight_cmd, is_dry_run=args.dry_run)

    print("\n" + "="*60 + f"\n🎉 Experiment '{config_name}' finished.\n" + "="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GIR experiments with specified configurations.")
    parser.add_argument("config", choices=EXPERIMENT_CONFIGS.keys(), help="Name of the experiment configuration to run.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    args = parser.parse_args()
    main(args)