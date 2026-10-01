import os
import shutil
from pathlib import Path

def main():
    # Use paths relative to the script location (REvolution directory)
    base_dir = Path(__file__).resolve().parent
    exp_dir = base_dir / "exp"
    out_dir = base_dir / "synthesizable"
    
    if not exp_dir.exists():
        print(f"Error: {exp_dir} does not exist.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    # Walk through exp_dir
    for root, dirs, files in os.walk(exp_dir):
        for f in files:
            # Check for synthesis output file. The prompt mentioned ".syn.sv", but actual files in REvolution/exp use ".syn.v"
            # Check both to be safe.
            if f.endswith('.syn.sv') or f.endswith('.syn.v'):
                # Determine original .sv file name
                if f.endswith('.syn.sv'):
                    orig_name = f[:-7] + ".sv"
                else:
                    orig_name = f[:-6] + ".sv"
                
                orig_path = Path(root) / orig_name
                
                if orig_path.exists():
                    # Calculate relative path to maintain directory structure
                    rel_path = orig_path.relative_to(exp_dir)
                    dest_path = out_dir / rel_path
                    
                    # Create parent directories
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Copy the original .sv file
                    shutil.copy2(orig_path, dest_path)
                    count += 1
    
    print(f"Successfully collected {count} synthesizable .sv files to {out_dir}")

if __name__ == "__main__":
    main()
