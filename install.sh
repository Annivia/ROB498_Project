#!/bin/bash

# Set formatting
reset="\033[0m"
green="\033[32m"
blue="\033[34m"
bold="\033[1m"

# Get the terminal width
terminal_width=$(tput cols)

# Generate the separator line
separator=$(printf '%*s' "$terminal_width" '' | tr ' ' '#')

echo -e "${bold}${green}${separator}${reset}"
echo -e "${bold}${green}This environment requires strictly Python 3.10 to install.${reset}"
echo -e "${bold}${blue}If your operating system / package manager is not supported, please install Python 3.10 manually.${reset}"
echo -e "${bold}${green}${separator}${reset}"


# Check if Python 3.10 is installed
python_version=$(python -c "import sys; print('{}.{}'.format(sys.version_info.major, sys.version_info.minor))")

if [ "$python_version" != "3.10" ]; then
  echo "Python 3.10 not found. Installing Python 3.10..."
  
  # Check the operating system
  os=$(uname)

  if [ "$os" == "Linux" ]; then
    # Check the package manager
    if command -v apt-get &> /dev/null; then
      # Install Python 3.10 for Debian/Ubuntu
      sudo apt-get install -y python3.10 python3.10-venv python3.10-distutils
    elif command -v pacman &> /dev/null; then
      # Install Python 3.10 for Arch Linux
      sudo pacman -S python310
    else
      echo "Unsupported package manager. Please install Python 3.10 manually."
      exit 1
    fi
  elif [ "$os" == "Darwin" ]; then
    # Install Python 3.10 for macOS using Homebrew
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
    brew install python@3.10
  else
    echo "Unsupported operating system. Please install Python 3.10 manually."
    exit 1
  fi
fi

# Install packages from requirements.txt
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
