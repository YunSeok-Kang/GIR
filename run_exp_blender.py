import subprocess
import os
import glob
import argparse
import shlex

# ==============================================================================
# ⚙️ 설정 (경로가 다르다면 이 부분을 수정하세요)
# ==============================================================================

# 1. NeRF 데이터셋의 기본 경로
base_scene_dir = "nerf_synthetic"

# 2. HDR 이미지가 있는 경로
hdri_dir = "hdri"

# 3. 모든 결과물이 저장될 기본 출력 디렉토리
output_dir = "output_blender"

# ==============================================================================

def run_command(command, is_dry_run=False):
    """주어진 명령어를 실행하거나, dry run 모드에서는 출력만 합니다."""
    # 리스트 형태의 명령어를 터미널에서 바로 복사/실행 가능한 문자열로 변환
    command_str = ' '.join(shlex.quote(str(arg)) for arg in command)
    
    if is_dry_run:
        # Dry Run 모드일 경우, 실행될 명령어만 출력
        print(f"🐫 [DRY RUN] Would execute: {command_str}")
        return

    # 실제 실행 모드
    print(f"\n▶️ Executing: {command_str}")
    try:
        # 터미널 환경에서는 check=True로 간단하게 실행
        subprocess.run(command, check=True, text=True)
        print("✅ Command finished successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR: Command failed with exit code {e.returncode}")
        # 오류가 발생하면 스크립트 실행을 중단시킴
        raise e
    except FileNotFoundError:
        print(f"❌ ERROR: Command '{command[0]}' not found. Is the environment set up correctly?")
        raise

def main(args):
    """모든 장면과 HDR 조합에 대한 실험을 자동으로 실행합니다."""
    
    if args.dry_run:
        print("\n" + "#"*60)
        print("🐫 DRY RUN MODE ENABLED. NO COMMANDS WILL BE EXECUTED. 🐫")
        print("#"*60 + "\n")
    
    # 결과물 폴더 생성
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. 실험할 장면과 HDR 목록 자동 탐색 ---
    print("📂 1. Scanning for scenes and HDR files...")
    try:
        scenes_to_process = [d for d in os.listdir(base_scene_dir) if os.path.isdir(os.path.join(base_scene_dir, d))]
        hdrs_to_process = glob.glob(os.path.join(hdri_dir, '*.hdr'))
    except FileNotFoundError as e:
        print(f"❌ ERROR: Directory not found - {e.filename}. Please check the paths in the script.")
        return

    if not scenes_to_process or not hdrs_to_process:
        print("❌ Error: No scenes or HDR files found. Please check the paths in the script.")
        return
        
    print(f"Found {len(scenes_to_process)} scenes: {scenes_to_process}")
    print(f"Found {len(hdrs_to_process)} HDR images: {[os.path.basename(p) for p in hdrs_to_process]}")

    # --- 2. 모든 조합에 대해 실험 실행 ---
    total_experiments = len(scenes_to_process) * len(hdrs_to_process)
    current_experiment = 0

    for scene_name in scenes_to_process:
        print("\n" + "="*60)
        print(f"🚀 Processing Scene: [{scene_name}]")
        print("="*60)

        # --- 2a. 모델 훈련 (장면 당 한 번만 실행) ---
        scene_path = os.path.join(base_scene_dir, scene_name)
        model_output_folder = f"{scene_name}_model"
        model_path = os.path.join(output_dir, model_output_folder)
        
        train_cmd = ["python", "train.py", "-s", scene_path, "-m", model_path, "--eval", "--random_background", "--hdr_rotation"]
        run_command(train_cmd, is_dry_run=args.dry_run)

        # --- 2b. Relight 렌더링 ---
        for hdr_path in hdrs_to_process:
            current_experiment += 1
            print("\n" + "-"*50)
            print(f"💡 Applying HDR [{os.path.basename(hdr_path)}] to [{scene_name}] ({current_experiment}/{total_experiments})")
            
            hdr_basename = os.path.splitext(os.path.basename(hdr_path))[0]
            relight_save_name = f"{scene_name}_{hdr_basename}"

            relight_cmd = [
                "python", "render.py",
                "-m", model_path,
                "--skip_train",
                "--save_name", relight_save_name,
                "-w", "--hdr_rotation",
                "--environment_texture", hdr_path,
                "--render_relight"
            ]
            run_command(relight_cmd, is_dry_run=args.dry_run)
    
    print("\n" + "="*60)
    if not args.dry_run:
        print("🎉 All experiments completed successfully!")
    else:
        print("🐫 Dry run finished. All commands have been listed.")
    print("="*60)

if __name__ == "__main__":
    # --- Dry Run을 위한 Argument Parser 설정 ---
    parser = argparse.ArgumentParser(description="Run GIR experiments for all scenes and HDRs.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would be executed without actually running them."
    )
    args = parser.parse_args()
    main(args)