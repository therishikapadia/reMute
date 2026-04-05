import subprocess
import os
import sys

def main():
    """Launch the Streamlit dashboard."""
    print("🚀 Starting reMute Dashboard...")
    
    # Use the current python executable to run streamlit as a module
    # This is more robust on macOS and avoids SIGTRAP/permission errors
    cmd = [sys.executable, "-m", "streamlit", "run", "ui/app.py"]

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 reMute stopped.")
    except Exception as e:
        print(f"❌ Error launching Streamlit: {e}")
        print("💡 Try running: source venv/bin/activate && streamlit run ui/app.py")

if __name__ == "__main__":
    main()
