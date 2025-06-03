#!/usr/bin/env python3
"""
Setup script for Meme OCR system.

This script helps install dependencies and set up the environment.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command):
    """Run a shell command and return success status."""
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr


def check_tesseract():
    """Check if Tesseract OCR is installed."""
    success, output = run_command("tesseract --version")
    return success


def install_tesseract():
    """Install Tesseract OCR based on the platform."""
    print("Installing Tesseract OCR...")
    
    # Detect platform
    import platform
    system = platform.system().lower()
    
    if system == "darwin":  # macOS
        print("Detected macOS. Installing via Homebrew...")
        success, output = run_command("brew install tesseract")
        if not success:
            print("Failed to install via Homebrew. Please install manually:")
            print("brew install tesseract")
            return False
    
    elif system == "linux":
        print("Detected Linux. Installing via apt...")
        success, output = run_command("sudo apt-get update && sudo apt-get install -y tesseract-ocr")
        if not success:
            print("Failed to install via apt. Please install manually:")
            print("sudo apt-get install tesseract-ocr")
            return False
    
    else:
        print(f"Unsupported platform: {system}")
        print("Please install Tesseract OCR manually:")
        print("Windows: https://github.com/UB-Mannheim/tesseract/wiki")
        print("Linux: sudo apt-get install tesseract-ocr")
        print("macOS: brew install tesseract")
        return False
    
    return True


def install_python_deps():
    """Install Python dependencies."""
    print("Installing Python dependencies...")
    
    # Check if we're in a virtual environment
    if not hasattr(sys, 'real_prefix') and sys.base_prefix == sys.prefix:
        print("Warning: Not in a virtual environment. Consider using 'source venv/bin/activate'")
    
    success, output = run_command("pip install -r requirements.txt")
    return success


def main():
    print("=== Meme OCR Setup ===")
    
    # Check current directory
    if not Path("requirements.txt").exists():
        print("Error: requirements.txt not found. Please run this script from the project directory.")
        sys.exit(1)
    
    # Check Tesseract
    print("Checking Tesseract OCR...")
    if not check_tesseract():
        print("Tesseract OCR not found.")
        response = input("Install Tesseract OCR? (y/n): ").lower().strip()
        if response == 'y':
            if not install_tesseract():
                print("Failed to install Tesseract. Please install manually and try again.")
                sys.exit(1)
        else:
            print("Tesseract OCR is required. Please install it manually.")
            sys.exit(1)
    else:
        print("✓ Tesseract OCR found")
    
    # Install Python dependencies
    print("\nInstalling Python dependencies...")
    if install_python_deps():
        print("✓ Python dependencies installed")
    else:
        print("Failed to install Python dependencies. Please check your environment.")
        sys.exit(1)
    
    print("\n=== Setup Complete! ===")
    print("\nNext steps:")
    print("1. Extract text from memes: python meme_ocr.py --extract")
    print("2. Search memes: python meme_ocr.py --search 'your search term'")
    print("3. View stats: python meme_ocr.py --stats")


if __name__ == "__main__":
    main()