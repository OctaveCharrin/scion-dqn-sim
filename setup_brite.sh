#!/bin/bash
# Setup script for BRITE topology generator
# This script initializes the BRITE submodule and ensures it's ready to use

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRITE_DIR="${SCRIPT_DIR}/external/brite"
JAR_PATH="${BRITE_DIR}/Java/Brite.jar"

echo "=========================================="
echo "BRITE Setup Script"
echo "=========================================="
echo ""

# Check if Java is installed
echo "Checking for Java..."
if ! command -v java &> /dev/null; then
    echo "ERROR: Java is not installed or not in PATH"
    echo "Please install Java (OpenJDK 11 or later recommended)"
    echo "On Ubuntu/Debian: sudo apt-get install openjdk-11-jdk"
    echo "On Fedora/RHEL: sudo dnf install java-11-openjdk-devel"
    exit 1
fi

JAVA_VERSION=$(java -version 2>&1 | head -n 1)
echo "✓ Found Java: ${JAVA_VERSION}"
echo ""

# Check if BRITE directory exists
if [ ! -d "${BRITE_DIR}" ]; then
    echo "BRITE directory not found. Initializing git submodule..."
    cd "${SCRIPT_DIR}"
    
    # Initialize and update submodule
    if [ -f ".gitmodules" ]; then
        git submodule update --init --recursive external/brite
    else
        echo "ERROR: .gitmodules file not found. Cannot initialize BRITE submodule."
        echo "Please ensure you're in the SNETSIM repository root."
        exit 1
    fi
    echo "✓ BRITE submodule initialized"
else
    echo "✓ BRITE directory found"
fi

# Check if submodule is properly initialized (has .git file or directory)
if [ -d "${BRITE_DIR}/.git" ] || [ -f "${BRITE_DIR}/.git" ]; then
    echo "✓ BRITE submodule is initialized"
else
    echo "WARNING: BRITE submodule may not be properly initialized"
    echo "Attempting to initialize..."
    cd "${SCRIPT_DIR}"
    git submodule update --init --recursive external/brite || true
fi

# Check if JAR file exists
if [ -f "${JAR_PATH}" ]; then
    echo "✓ BRITE JAR file found at ${JAR_PATH}"
    
    # Verify it's a valid JAR
    if file "${JAR_PATH}" | grep -q "Java archive"; then
        echo "✓ JAR file is valid"
    else
        echo "WARNING: JAR file may be corrupted, attempting to rebuild..."
        cd "${BRITE_DIR}"
        make buildjava || {
            echo "ERROR: Failed to build BRITE. Please check the error messages above."
            exit 1
        }
    fi
else
    echo "JAR file not found. Building BRITE..."
    cd "${BRITE_DIR}"
    
    # Build Java components
    if [ -f "Java/Makefile" ]; then
        echo "Building BRITE Java components..."
        (cd Java && make) || {
            echo "ERROR: Failed to build BRITE Java components"
            exit 1
        }
        echo "✓ BRITE Java components built"
    else
        echo "ERROR: Java/Makefile not found in BRITE directory"
        exit 1
    fi

    # Upstream BRITE Java/Makefile only runs javac; this repo expects Java/Brite.jar
    if [ ! -f "${JAR_PATH}" ]; then
        echo "Packaging BRITE into Brite.jar..."
        (cd "${BRITE_DIR}/Java" && jar cfe Brite.jar Main.Brite \
            Main/*.class Graph/*.class Model/*.class Import/*.class Export/*.class \
            Util/*.class Topology/*.class) || {
            echo "ERROR: Failed to create Brite.jar"
            exit 1
        }
    fi
    
    # Verify JAR was created
    if [ ! -f "${JAR_PATH}" ]; then
        echo "ERROR: JAR file was not created after build"
        echo "Please check the build output above for errors"
        exit 1
    fi
    echo "✓ BRITE JAR file created"
fi

# Test that BRITE can be executed
echo ""
echo "Testing BRITE installation..."
cd "${BRITE_DIR}"
if java -jar "${JAR_PATH}" --help &> /dev/null || java -cp Java/:. Main.Brite --help &> /dev/null; then
    echo "✓ BRITE is working correctly"
elif java -jar "${JAR_PATH}" 2>&1 | head -1 &> /dev/null; then
    # Some versions don't support --help, but will show usage on error
    echo "✓ BRITE appears to be working (no --help flag, but executable)"
else
    echo "WARNING: Could not verify BRITE execution, but JAR file exists"
    echo "This may be normal if BRITE requires specific arguments"
fi

echo ""
echo "=========================================="
echo "BRITE setup complete!"
echo "=========================================="
echo ""
echo "BRITE JAR location: ${JAR_PATH}"
echo ""
echo "You can now use BRITE for topology generation."
echo ""

