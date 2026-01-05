#!/bin/bash
# CARLA UE5 (0.10.0) Setup Script for Ubuntu

set -e

CARLA_VERSION="0.10.0"
INSTALL_DIR="$HOME/carla_ue5"

echo "=============================================="
echo "CARLA UE5 ($CARLA_VERSION) Setup Script"
echo "=============================================="
echo ""
echo "System Requirements:"
echo "  - Ubuntu 22.04+ (recommended)"
echo "  - NVIDIA GPU with Vulkan support"
echo "  - ~100GB disk space"
echo "  - 16GB+ RAM"
echo ""

# Check disk space
AVAILABLE_GB=$(df -BG ~ | tail -1 | awk '{print $4}' | tr -d 'G')
echo "Available disk space: ${AVAILABLE_GB}GB"
if [ "$AVAILABLE_GB" -lt 100 ]; then
    echo "WARNING: Less than 100GB available. CARLA UE5 may not fit."
fi

# Check for Vulkan
echo ""
echo "Checking Vulkan support..."
if command -v vulkaninfo &> /dev/null; then
    echo "Vulkan found: $(vulkaninfo --summary 2>/dev/null | grep 'deviceName' | head -1 || echo 'Vulkan available')"
else
    echo "WARNING: vulkaninfo not found. Install vulkan-tools:"
    echo "  sudo apt install vulkan-tools"
fi

echo ""
echo "=============================================="
echo "Download Options:"
echo "=============================================="
echo ""
echo "Option 1: Pre-built Package (Recommended)"
echo "  Download from: https://github.com/carla-simulator/carla/releases"
echo "  Look for: CARLA_0.10.0_Linux.tar.gz (or latest UE5 version)"
echo ""
echo "Option 2: Build from Source"
echo "  git clone https://github.com/carla-simulator/carla.git"
echo "  cd carla && git checkout ue5-dev"
echo "  make setup && make build"
echo ""
echo "=============================================="
echo "After Download:"
echo "=============================================="
echo ""
echo "1. Extract to $INSTALL_DIR:"
echo "   mkdir -p $INSTALL_DIR"
echo "   tar -xvf CARLA_0.10.0_Linux.tar.gz -C $INSTALL_DIR"
echo ""
echo "2. Run CARLA UE5 server:"
echo "   cd $INSTALL_DIR"
echo "   ./CarlaUnreal.sh -vulkan -RenderOffScreen"
echo ""
echo "3. Install Python API:"
echo "   pip install carla==0.10.0"
echo ""
echo "4. Collect data:"
echo "   python collect_data_ue5.py --port 2000 --output ./data_ue5/"
echo ""
echo "=============================================="
echo "Key Differences from CARLA 0.9.x (UE4):"
echo "=============================================="
echo "  - Vulkan rendering (better performance)"
echo "  - Improved lighting and shadows"
echo "  - New maps (Town15, etc.)"
echo "  - Higher resolution textures"
echo "  - API mostly compatible with 0.9.x"
echo ""
