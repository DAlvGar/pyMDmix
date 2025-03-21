#!/bin/bash

# Script to run the GROMACS hybrid approach tests

echo "Running GROMACS hybrid approach tests for multi-chain proteins"

# Ensure we're in the right directory
cd "$(dirname "$0")"

# Check if pytest is available (preferred for modern Python testing)
if command -v pytest > /dev/null 2>&1; then
    echo "Running tests with pytest..."
    pytest -v pyMDMix/test_gromacs_hybrid.py
else
    echo "pytest not found, using unittest directly..."
    python -m pyMDMix.test_gromacs_hybrid
fi

# Create a directory for more comprehensive testing with real data
mkdir -p test_data/gromacs_hybrid
cd test_data/gromacs_hybrid

echo "To run comprehensive tests with real data, place AMBER files in the test_data/gromacs_hybrid directory:"
echo "1. A multi-chain protein (test_multichain.prmtop, test_multichain.prmcrd, test_multichain.pdb)"
echo "2. A protein with missing loops (test_missingloop.prmtop, test_missingloop.prmcrd, test_missingloop.pdb)"
echo "3. Run the tests again with: FULL_TEST=1 ./run_gromacs_tests.sh"

# Check if we should run the comprehensive tests
if [ "$FULL_TEST" = "1" ] && [ -f "test_multichain.prmtop" ]; then
    echo "Running comprehensive tests with real data..."
    # This would be implemented to use actual data files for real-world testing
    python -c "
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pyMDMix.GROMACS import GROMACSWriter
from pyMDMix.test_gromacs_hybrid import MockReplica, MockSystem

# Create a test with real data
class RealReplica(MockReplica):
    def __init__(self, name, top, crd, ref=None):
        super().__init__(name, top, crd, ref)
        # Use actual file paths
        self.top = 'test_data/gromacs_hybrid/' + top
        self.crd = 'test_data/gromacs_hybrid/' + crd
        self.ref = 'test_data/gromacs_hybrid/' + ref if ref else None
        self.grotop = self.top.replace('prmtop', 'top')
        self.gro = self.crd.replace('prmcrd', 'gro')

# Test with multichain protein
print('Testing with multichain protein')
replica = RealReplica('test_multi', 'test_multichain.prmtop', 'test_multichain.prmcrd', 'test_multichain.pdb')
gromacs = GROMACSWriter()
gromacs.replica = replica
gromacs.convertAmberToGromacs()

# Test with missing loops
print('Testing with protein with missing loops')
replica = RealReplica('test_loop', 'test_missingloop.prmtop', 'test_missingloop.prmcrd', 'test_missingloop.pdb')
gromacs = GROMACSWriter()
gromacs.replica = replica
gromacs.convertAmberToGromacs()

print('Comprehensive tests completed successfully')
"
fi

echo "Tests completed" 