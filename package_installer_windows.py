import sys
import subprocess

def install_package(package):
    """Install or update the package using pip."""
    try:
        print(f"Installing/Updating package: {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package])
        print(f"Successfully installed/updated {package}")
    except subprocess.CalledProcessError:
        print(f"Failed to install/update {package}")

def check_and_install_packages(packages):
    """Check if packages are installed, and install/update them if needed."""
    for package in packages:
        if not is_package_installed(package):
            print(f"Package '{package}' is not installed. Installing...")
            install_package(package)
        else:
            print(f"Package '{package}' is already installed.")

def is_package_installed(package):
    """Check if a package is installed by running 'pip show'."""
    try:
        # Run 'pip show' to check if the package is installed
        subprocess.check_call([sys.executable, "-m", "pip", "show", package], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False

if __name__ == "__main__":
    # List of packages you want to install/update
    packages = ['diffusers', 'gdown', 'ipython', 'numpy', 'pandas', 'Pillow', 'requests', 'streamlit', 'torch']
    
    # Call the function to check and install/update the packages
    check_and_install_packages(packages)
