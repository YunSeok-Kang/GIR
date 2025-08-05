# 파일명: run_exp_blender.py
import subprocess
import os
import glob
import argparse
import shlex

# ==============================================================================
# ⚙️ 설정 (경로가 다르다면 이 부분을 수정하세요)
# ==============================================================================
base_scene_dir = "nerf_synthetic"
hdri_dir = "hdri"
output_dir = "output_blender"

# ==============================================================================
# run_command 함수는 이전과 동일
def run_command(command, is_dry_run=False):
    """주어진 명령어를 실행하거나, dry run 모드에서는 출력만 합니다."""
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
    """모든 장면과 HDR 조합에 대한 실험을 자동으로 실행합니다."""
    if args.dry_run:
        print("\n" + "#"*60 + "\n🐫 DRY RUN MODE ENABLED\n" + "#"*60 + "\n")
    os.makedirs(output_dir, exist_ok=True)

    print("📂 1. Scanning for scenes and HDR files...")
    try:
        scenes_to_process = [d for d in os.listdir(base_scene_dir) if os.path.isdir(os.path.join(base_scene_dir, d))]
        hdrs_to_process = glob.glob(os.path.join(hdri_dir, '*.hdr'))
    except FileNotFoundError as e:
        print(f"❌ ERROR: Directory not found - {e.filename}.")
        return

    print(f"Found {len(scenes_to_process)} scenes: {scenes_to_process}")
    print(f"Found {len(hdrs_to_process)} HDR images: {[os.path.basename(p) for p in hdrs_to_process]}")
    
    total_relight_experiments = len(scenes_to_process) * len(hdrs_to_process)
    current_relight_experiment = 0

    for scene_name in scenes_to_process:
        print("\n" + "="*60 + f"\n🚀 Processing Scene: [{scene_name}]\n" + "="*60)
        
        scene_path = os.path.join(base_scene_dir, scene_name)
        model_output_folder = f"{scene_name}_model"
        model_path = os.path.join(output_dir, model_output_folder)
        
        # --- 2a. 모델 훈련 (장면 당 한 번만 실행) ---
        print("\n" + "-"*50 + "\n🏋️ Training model...\n" + "-"*50)
        train_cmd = ["python", "train.py", "-s", scene_path, "-m", model_path, "--eval", "--random_background", "--hdr_rotation"]
        run_command(train_cmd, is_dry_run=args.dry_run)

        # --- ✨ 2b. 원본(Ground Truth) 비디오용 이미지 렌더링 (장면 당 한 번만 실행) ---
        print("\n" + "-"*50 + "\n🖼️ Rendering ground truth images for original video...\n" + "-"*50)
        # --render_relight 옵션 없이 render.py를 실행하면 원본(gt) 이미지가 저장됩니다.
        gt_save_name = "ground_truth" # 저장될 폴더 이름
        gt_render_cmd = ["python", "render.py", "-m", model_path, "--skip_train", "--save_name", gt_save_name]
        run_command(gt_render_cmd, is_dry_run=args.dry_run)
        
        # --- 2c. Relight 렌더링 (각 HDR에 대해 실행) ---
        for hdr_path in hdrs_to_process:
            current_relight_experiment += 1
            print("\n" + "-"*50)
            print(f"💡 Applying HDR [{os.path.basename(hdr_path)}] to [{scene_name}] ({current_relight_experiment}/{total_relight_experiments})")
            
            hdr_basename = os.path.splitext(os.path.basename(hdr_path))[0]
            relight_save_name = f"{scene_name}_{hdr_basename}"

            relight_cmd = ["python", "render.py", "-m", model_path, "--skip_train", "--save_name", relight_save_name, "-w", "--hdr_rotation", "--environment_texture", hdr_path, "--render_relight"]
            run_command(relight_cmd, is_dry_run=args.dry_run)
    
    print("\n" + "="*60 + "\n🎉 All rendering tasks are complete. You can now run create_videos.py.\n" + "="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GIR experiments for all scenes and HDRs.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    args = parser.parse_args()
    main(args)