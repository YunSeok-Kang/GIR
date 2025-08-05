# 파일명: create_videos.py
import subprocess
import os
import glob
import argparse
import shlex
import shutil

# ==============================================================================
# ⚙️ 실험 설정 (run_experiment.py와 동기화)
# ==============================================================================
EXPERIMENT_CONFIGS = {
    "blender_synthetic": {
        "output_dir": "output_blender",
        "videos_dir": "output_videos_blender"
    },
    "shiny_blender": {
        "output_dir": "output_shiny_blender",
        "videos_dir": "output_videos_shiny_blender"
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
        subprocess.run(command, check=True, text=True, capture_output=True)
        print("✅ Video created successfully.")
    except Exception as e:
        print(f"❌ ERROR: {e}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"   Error message: {e.stderr.strip()}")
        raise

def main(args):
    config_name = args.config
    config = EXPERIMENT_CONFIGS[config_name]
    base_output_dir = config["output_dir"]
    collected_videos_dir = config["videos_dir"]

    print(f"🎬 Creating and collecting videos for configuration: '{config_name}'")

    if args.dry_run:
        print("\n" + "#"*60 + "\n🐫 DRY RUN MODE ENABLED\n" + "#"*60 + "\n")
    else:
        os.makedirs(collected_videos_dir, exist_ok=True)
    
    search_pattern_relit = os.path.join(base_output_dir, "*", "test_*", "ours_*", "renders")
    search_pattern_gt = os.path.join(base_output_dir, "*", "test_ground_truth", "ours_*", "gt")
    image_folders = glob.glob(search_pattern_relit) + glob.glob(search_pattern_gt)

    if not image_folders:
        print("❌ No image sequences found to create videos.")
        return

    for image_dir in image_folders:
        parent_dir = os.path.dirname(image_dir)
        grandparent_dir = os.path.dirname(parent_dir)
        grandparent_dir_name = os.path.basename(grandparent_dir)
        
        if os.path.basename(image_dir) == "gt":
            great_grandparent_dir = os.path.dirname(grandparent_dir)
            model_folder_name = os.path.basename(great_grandparent_dir)
            scene_name = model_folder_name.replace('_model', '')
            video_name = f"test_{scene_name}_ground_truth.mp4"
        else:
            video_name = f"{grandparent_dir_name}.mp4"

        temp_video_path = os.path.join(parent_dir, video_name)
        print("\n" + "-"*50 + f"\nProcessing: {image_dir}")
        print(f"  -> Creating video: {video_name}")
        
        ffmpeg_cmd = ["ffmpeg", "-r", "30", "-i", os.path.join(image_dir, "%05d.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-y", temp_video_path]
        run_command(ffmpeg_cmd, is_dry_run=args.dry_run)

        if not args.dry_run:
            final_video_path = os.path.join(collected_videos_dir, video_name)
            print(f"  -> Moving video to: {final_video_path}")
            shutil.move(temp_video_path, final_video_path)

    print("\n" + "="*60 + f"\n🎉 Post-processing for '{config_name}' finished.\n" + "="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create and collect videos from experiment results.")
    parser.add_argument("config", choices=EXPERIMENT_CONFIGS.keys(), help="Name of the experiment configuration to process.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    args = parser.parse_args()
    main(args)