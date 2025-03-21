# GROMACS Multi-Chain Protein Support Improvement

## Overview

This update enhances the GROMACS implementation in pyMDMix to properly handle multi-chain proteins and proteins with missing loops. The previous implementation had limitations when dealing with non-contiguous protein structures, which caused problems during restraint generation.

## Implementation Details

The solution implements a hybrid approach that combines the strengths of:
- **ParmEd** for accurate conversion of AMBER parameters to GROMACS format
- **GROMACS pdb2gmx** with specific chain handling options for proper handling of missing loops
- **Custom topology merging** to maintain parameter integrity while fixing chain organization

### Key Features

1. **Chain Detection**: Automatically identifies proteins with multiple chains or missing loops
2. **Parameter Preservation**: Maintains all force field parameters from the original AMBER topology
3. **Proper Chain Handling**: Uses GROMACS's built-in chain handling capabilities
4. **Fallback Mechanism**: Gracefully falls back to standard conversion when needed
5. **Force Field Mapping**: Automatically maps AMBER force fields to GROMACS force fields

### New Methods Added

- `_checkMultipleChains()`: Detects if a protein has multiple chains or missing loops
- `_mapAmberToGromacsFF()`: Maps AMBER force field and water model to GROMACS equivalents
- `_convertWithHybridApproach()`: Implements the hybrid conversion strategy
- `_mergeChainInfoWithParameters()`: Merges chain information with original parameters

### Updated Methods

- `convertAmberToGromacs()`: Modified to use the hybrid approach when needed

## Testing

A comprehensive test suite has been added to validate the new implementation:

1. **Unit Tests**:
   - Test for multiple chains detection
   - Test for missing loops detection
   - Test for force field mapping
   - Test for topology merging

2. **Integration Tests**:
   - Mock test for the overall conversion process
   - Optional real data tests (requires actual AMBER files)

To run the tests:
```bash
./run_gromacs_tests.sh
```

For comprehensive testing with real data, place AMBER files in the test_data/gromacs_hybrid directory:
1. A multi-chain protein (test_multichain.prmtop, test_multichain.prmcrd, test_multichain.pdb)
2. A protein with missing loops (test_missingloop.prmtop, test_missingloop.prmcrd, test_missingloop.pdb)
3. Run the tests with: `FULL_TEST=1 ./run_gromacs_tests.sh`

## Benefits of the Hybrid Approach

1. **Accurate Parameters**: All force field parameters from AMBER are preserved exactly
2. **Robust Chain Handling**: Properly handles proteins with multiple chains or missing loops
3. **Simulation Consistency**: Ensures consistent simulations across different MD engines
4. **Future Compatibility**: Leverages standard GROMACS tools in a way that will remain compatible with future versions

## Usage

The implementation is transparent to users - no change in workflow is required. The system automatically detects when the hybrid approach is needed and applies it.

## Force Field Mapping

The implementation maps AMBER force fields to their GROMACS equivalents:

| AMBER Force Field | GROMACS Force Field |
|-------------------|---------------------|
| ff14SB            | amber14sb           |
| ff99SB            | amber99sb           |
| ff99SBildn        | amber99sb-ildn      |

And water models:

| AMBER Water Model | GROMACS Water Model |
|-------------------|---------------------|
| tip3p             | tip3p               |
| tip4p             | tip4p               |
| tip4pew           | tip4p-ew            |
| tip5p             | tip5p               |
| spce              | spce                | 