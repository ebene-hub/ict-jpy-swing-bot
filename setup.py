#!/usr/bin/env python3
"""
ICT Swing Trading AI - Setup Script
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check Python version compatibility"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("Python 3.8 or higher is required")
        return False
    print(f"Python {version.major}.{version.minor}.{version.micro} - Compatible")
    return True

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install requirements: {e}")
        return False

def setup_environment():
    """Setup environment variables"""
    print("Setting up environment...")

    # Create .env file if it doesn't exist
    if not os.path.exists(".env"):
        try:
            with open(".env.example", "r") as example_file:
                example_content = example_file.read()

            with open(".env", "w") as env_file:
                env_file.write(example_content)

            print("Created .env file from template")
            print("Please update .env with your MT5 credentials")
        except Exception as e:
            print(f"Failed to create .env file: {e}")
            return False
    else:
        print(".env file already exists")

    return True

def create_directories():
    """Create necessary directories"""
    print("Creating directories...")

    directories = [
        "logs",
        "results", 
        "exports",
        "data/cache"
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created {directory}")

    return True

def verify_mt5_installation():
    """Verify MT5 installation"""
    print("Checking MT5 installation...")

    system = platform.system()

    if system == "Windows":
        common_paths = [
            "C:\\Program Files\\Five Percent Online MetaTrader 5\\terminal64.exe",
            "C:\\Program Files (x86)\\MetaTrader 5\\terminal64.exe",
        ]

        for path in common_paths:
            if os.path.exists(path):
                print(f"MT5 found at: {path}")
                return True

        print("MT5 not found in common locations")
        print("Please ensure MT5 is installed and update the path in .env")
        return False

    else:
        print("MT5 is only available on Windows")
        return False

def run_tests():
    """Run basic tests"""
    print("Running basic tests...")

    try:
        # Test basic imports
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt

        print("Core libraries imported successfully")

        # Test project imports
        sys.path.append('.')
        from data.data_loader import MT5DataLoader
        from strategies.ict_strategy import ICTSwingStrategy

        print("Project modules imported successfully")

        return True

    except ImportError as e:
        print(f"Import test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("ICT Swing Trading AI - Setup")
    print("="*50)

    steps = [
        ("Python Version Check", check_python_version),
        ("Requirements Installation", install_requirements),
        ("Environment Setup", setup_environment),
        ("Directory Creation", create_directories),
        ("MT5 Verification", verify_mt5_installation),
        ("Basic Tests", run_tests),
    ]

    all_passed = True

    for step_name, step_function in steps:
        print(f"{step_name}...")
        if not step_function():
            all_passed = False
            print(f"{step_name} failed")
        else:
            print(f"{step_name} completed")

    print("="*50)
    if all_passed:
        print("Setup completed successfully!")
        print("\nNext steps:")
        print("1. Update .env with your MT5 credentials")
        print("2. Run: python main.py")
        print("3. Or open analysis.ipynb for interactive analysis")
    else:
        print("Setup completed with errors")
        print("Please fix the issues above and run setup again")

    print("="*50)

if __name__ == "__main__":
    main()
