#!/bin/bash

# Define the key and repository
INTEL_GPG_KEY_URL="https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB"
INTEL_REPO_URL="https://apt.repos.intel.com/oneapi"
KEYRING_PATH="/usr/share/keyrings/intel-oneapi-keyring.gpg"
REPO_LIST_FILE="/etc/apt/sources.list.d/oneapi.list"

# Download and install the GPG key in a separate keyring
echo "Downloading and installing the Intel GPG key..."
wget -qO- "$INTEL_GPG_KEY_URL" | gpg --dearmor | sudo tee "$KEYRING_PATH" >/dev/null

# Add the Intel oneAPI repository if it's not already present
if ! grep -Rq "^deb .*oneapi" /etc/apt/sources.list*; then
    echo "Adding the Intel oneAPI repository..."
    echo "deb [signed-by=$KEYRING_PATH] $INTEL_REPO_URL all main" | sudo tee "$REPO_LIST_FILE"
else
    echo "Intel oneAPI repository already exists."
fi

# Update apt package lists
echo "Updating apt package lists..."
sudo apt update

# Install the Intel oneAPI Base Toolkit
echo "Installing the Intel oneAPI Base Toolkit..."
sudo apt install intel-basekit

# Append the source command to /etc/profile if not already present
if ! grep -q "source /opt/intel/oneapi/setvars.sh" /etc/profile; then
    echo "Appending source command to /etc/profile for all users..."
    echo "sh /opt/intel/oneapi/setvars.sh" | sudo tee -a /etc/profile >/dev/null
else
    echo "The source command is already in /etc/profile."
fi

echo "Installation complete. Please sudo reboot now."

