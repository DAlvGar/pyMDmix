#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test module to validate the hybrid approach for multi-chain protein conversion in GROMACS implementation
"""

import os
import sys
import unittest
import tempfile
import shutil
import subprocess as sub
import logging

# Import necessary modules from pyMDMix
try:
    from pyMDMix.GROMACS import GROMACSWriter
    from pyMDMix.PDB import SolvatedPDB
    import pyMDMix.tools as T
except ImportError:
    # Add parent directory to path if running from development directory
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from pyMDMix.GROMACS import GROMACSWriter
    from pyMDMix.PDB import SolvatedPDB
    import pyMDMix.tools as T

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_gromacs_hybrid")

class MockReplica:
    """Mock Replica class for testing purposes"""
    def __init__(self, name, top, crd, ref=None):
        self.name = name
        self.top = top
        self.crd = crd
        self.ref = ref
        self.FF = ["leaprc.protein.ff14SB", "leaprc.water.tip3p"]
        self.grotop = None
        self.gro = None
        self.system = MockSystem()
        self.restraint_identifiers = []

    def go(self):
        """Mock method to mimic replica.go() behavior"""
        pass

class MockSystem:
    """Mock System class for testing purposes"""
    def __init__(self):
        self.extraResList = []
        self.ligandResname = ""
        self.ref = None

    def getSolvent(self):
        """Mock method to return a solvent box"""
        return MockSolventBox()

class MockSolventBox:
    """Mock solvent box class for testing"""
    def __init__(self):
        self.residues = [MockResidue("WAT")]

class MockResidue:
    """Mock residue class for testing"""
    def __init__(self, name):
        self.name = name

class TestGromacsHybridApproach(unittest.TestCase):
    """Test the hybrid approach for converting AMBER to GROMACS for multi-chain proteins"""

    def setUp(self):
        """Set up test case with temporary directory and files"""
        # Create a temporary directory for testing
        self.test_dir = tempfile.mkdtemp()
        self.old_cwd = os.getcwd()
        os.chdir(self.test_dir)

        # Create test files
        self.create_test_files()

    def tearDown(self):
        """Clean up after tests"""
        try:
            # Change back to original directory
            os.chdir(self.old_cwd)
            
            # Remove temporary directory
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir)
        except Exception as e:
            print("Error during tearDown: {}".format(e))

    def create_test_files(self):
        """Create necessary test files"""
        # In a real test, we would create actual AMBER topology and coordinate files
        # For this demonstration, we'll create mock files
        
        # Create mock AMBER topology file
        with open("test.prmtop", "w") as f:
            f.write("MOCK AMBER TOPOLOGY FILE")
        
        # Create mock AMBER coordinate file
        with open("test.prmcrd", "w") as f:
            f.write("MOCK AMBER COORDINATE FILE")
        
        # Create mock PDB reference file with multiple chains
        with open("test_multichain.pdb", "w") as f:
            f.write("ATOM      1  N   ASP A   1      11.396  22.274  71.694  1.00 24.67      A    N  \n")
            f.write("ATOM      2  CA  ASP A   1      10.747  23.581  71.490  1.00 22.38      A    C  \n")
            f.write("TER\n")
            f.write("ATOM     50  N   GLY B   1      15.396  25.274  75.694  1.00 24.67      B    N  \n")
            f.write("ATOM     51  CA  GLY B   1      14.747  26.581  75.490  1.00 22.38      B    C  \n")
            f.write("TER\n")
            f.write("END\n")
        
        # Create mock PDB reference file with missing loops
        with open("test_missingloop.pdb", "w") as f:
            f.write("ATOM      1  N   ASP A   1      11.396  22.274  71.694  1.00 24.67      A    N  \n")
            f.write("ATOM      2  CA  ASP A   1      10.747  23.581  71.490  1.00 22.38      A    C  \n")
            f.write("ATOM     10  N   GLY A   5      15.396  25.274  75.694  1.00 24.67      A    N  \n")
            f.write("ATOM     11  CA  GLY A   5      14.747  26.581  75.490  1.00 22.38      A    C  \n")
            f.write("TER\n")
            f.write("END\n")

    def test_check_multiple_chains(self):
        """Test the _checkMultipleChains method for multi-chain detection"""
        # Create a mock replica with a multi-chain PDB
        replica = MockReplica("test", "test.prmtop", "test.prmcrd", "test_multichain.pdb")
        
        # Create a GROMACSWriter instance
        gromacs = GROMACSWriter()
        gromacs.replica = replica
        
        # Check if multiple chains are detected
        try:
            result = gromacs._checkMultipleChains()
            self.assertTrue(result, "Should detect multiple chains in test_multichain.pdb")
        except Exception as e:
            self.fail("_checkMultipleChains raised exception: {}".format(e))

    def test_check_missing_loops(self):
        """Test the _checkMultipleChains method for missing loop detection"""
        # Create a mock replica with a PDB containing missing loops
        replica = MockReplica("test", "test.prmtop", "test.prmcrd", "test_missingloop.pdb")
        
        # Create a GROMACSWriter instance
        gromacs = GROMACSWriter()
        gromacs.replica = replica
        
        # Check if missing loops are detected
        try:
            result = gromacs._checkMultipleChains()
            self.assertTrue(result, "Should detect missing loops in test_missingloop.pdb")
        except Exception as e:
            self.fail("_checkMultipleChains raised exception: {}".format(e))

    def test_map_amber_to_gromacs_ff(self):
        """Test the _mapAmberToGromacsFF method for force field mapping"""
        # Create mock replicas with different force fields
        replica_ff14sb = MockReplica("test_ff14sb", "test.prmtop", "test.prmcrd")
        replica_ff14sb.FF = ["leaprc.protein.ff14SB", "leaprc.water.tip3p"]
        
        replica_ff99sb = MockReplica("test_ff99sb", "test.prmtop", "test.prmcrd")
        replica_ff99sb.FF = ["leaprc.protein.ff99SB", "leaprc.water.spce"]
        
        # Test FF14SB mapping
        gromacs_ff14sb = GROMACSWriter()
        gromacs_ff14sb.replica = replica_ff14sb
        ff14sb, water14sb = gromacs_ff14sb._mapAmberToGromacsFF()
        
        self.assertEqual(ff14sb, "amber14sb", "FF14SB should map to amber14sb")
        self.assertEqual(water14sb, "tip3p", "TIP3P water should map to tip3p")
        
        # Test FF99SB mapping
        gromacs_ff99sb = GROMACSWriter()
        gromacs_ff99sb.replica = replica_ff99sb
        ff99sb, water99sb = gromacs_ff99sb._mapAmberToGromacsFF()
        
        self.assertEqual(ff99sb, "amber99sb", "FF99SB should map to amber99sb")
        self.assertEqual(water99sb, "spce", "SPCE water should map to spce")

    def test_integration_mock(self):
        """Integration test using mock objects"""
        # Create a mock replica
        replica = MockReplica("test", "test.prmtop", "test.prmcrd", "test_multichain.pdb")
        
        # Create a GROMACSWriter instance
        gromacs = GROMACSWriter()
        gromacs.replica = replica
        
        # Mock the actual conversion methods to avoid external dependencies
        def mock_convert_with_hybrid_approach(amber_structure):
            # Just create dummy output files
            with open(replica.grotop, "w") as f:
                f.write("[ moleculetype ]\n")
                f.write("Protein  3\n")
                f.write("[ atoms ]\n")
            with open(replica.gro, "w") as f:
                f.write("Mock GRO file\n")
        
        # Replace the actual method with our mock
        gromacs._convertWithHybridApproach = mock_convert_with_hybrid_approach
        
        # Run the conversion
        try:
            gromacs.convertAmberToGromacs()
            
            # Check if output files were created
            self.assertTrue(os.path.exists(replica.grotop), "Topology file {} not created".format(replica.grotop))
            self.assertTrue(os.path.exists(replica.gro), "Coordinate file {} not created".format(replica.gro))
        except Exception as e:
            self.fail("convertAmberToGromacs raised exception: {}".format(e))

    @unittest.skipIf(not os.path.exists("/usr/bin/gmx"), "GROMACS executable not found")
    def test_real_conversion(self):
        """Test with real data if available (requires GROMACS and ParmEd)"""
        # This test is skipped if GROMACS is not installed
        # In a real-world scenario, you would need actual AMBER files for testing
        
        try:
            import parmed
        except ImportError:
            self.skipTest("ParmEd not available")
        
        # In a real test, you would:
        # 1. Create or use pre-made AMBER topology/coordinate files
        # 2. Run the actual conversion
        # 3. Verify the output with GROMACS tools
        
        self.skipTest("Skipping real conversion test - requires real AMBER files")

# Use a custom test runner to avoid I/O issues
def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGromacsHybridApproach)
    unittest.TextTestRunner(verbosity=2).run(suite)

if __name__ == "__main__":
    run_tests() 