# Development Plan: Improved Multi-Chain Protein Support for GROMACS in pyMDMix

## 1. Problem Statement

The current GROMACS implementation in pyMDMix has a significant limitation when handling proteins with multiple chains. This issue becomes particularly problematic for systems where loops are missing, causing the protein to be split into different segments that GROMACS interprets as separate chains. The current restraint application mechanism fails in these scenarios because:

1. The restraint generation process assumes a contiguous protein structure
2. The chain identification and separation logic doesn't properly handle fragmented proteins
3. The restraint ITP file creation doesn't account for complex multi-chain topologies

## 2. Current Implementation Analysis

### 2.1 Restraint Generation Workflow

The current workflow for restraint generation in the GROMACS implementation follows these steps:

1. `createGroups()` method creates index groups including a "protein" group
2. `createRestraints()` method:
   - Creates a partial GRO file for "protein" using `createPartialGROFromComplex()`
   - Generates a restraint ITP file using `createRestraintsITP()`
   - Does the same for extra residues and ligands
3. `createRestraintsITP()` method:
   - Uses `genrestr` to create restraint files
   - Modifies the topology to include these restraint files

The problem occurs primarily in the `createPartialGROFromComplex()` method that extracts chain segments using GROMACS selection syntax, which doesn't properly handle disconnected protein segments.

### 2.2 Root Causes

1. **Chain identification**: The current implementation uses standard GROMACS selections like "protein", which doesn't guarantee proper handling of fragmented proteins
2. **Group selection**: The `gmx make_ndx` selection syntax is not handling complex protein topologies properly
3. **Restraint generation**: The `gmx genrestr` tool doesn't properly generate restraints for disconnected chains as a unified entity

## 3. Proposed Solution: Hybrid Approach with ParmEd and pdb2gmx

After analyzing the issue, we propose implementing a hybrid approach that preserves all AMBER topology parameters while fixing chain handling issues. This solution leverages:

1. **ParmEd** for accurate conversion of AMBER parameters to GROMACS format
2. **GROMACS pdb2gmx** with specific chain handling options for proper handling of missing loops
3. **Custom merging of topologies** to maintain parameter integrity while fixing chain organization

This approach addresses the root cause of the problem by ensuring that missing loops don't create separate chains in the topology, while also preserving all force field parameters from the original AMBER files.

The key components of this solution are:
1. Detecting multi-chain proteins or proteins with missing loops
2. Converting AMBER files to GROMACS format using ParmEd to preserve parameters
3. Using pdb2gmx with `-chainsep id_and_ter` and `-merge all` options to handle chain issues
4. Merging the moleculetype and system definitions from pdb2gmx with the parameters from ParmEd
5. Ensuring position restraints work correctly with the merged topology

## 4. Implementation Plan

**File to modify**: `pyMDMix/GROMACS.py`

**Changes**:
1. Modify the `convertAmberToGromacs()` method to use a hybrid approach that preserves AMBER parameters while fixing chain handling:

```python
def convertAmberToGromacs(self, replica=False):
    """
    Convert AMBER files to GROMACS format using a hybrid approach that preserves
    all AMBER topology parameters while ensuring proper chain handling.
    """
    import parmed as pmd
    import tempfile
    import os
    import subprocess as sub
    
    replica = replica or self.replica
    if not replica: raise GROMACSWriterError, "Replica not assigned."
    
    # Load the AMBER prmtop and inpcrd files
    self.log.info("Converting Amber TOP/CRD to GROMACS format for replica %s ..."%(replica.name))
    
    # Define output filenames
    self.replica.grotop = self.replica.top.replace('prmtop','top')
    self.replica.gro = self.replica.crd.replace('prmcrd','gro')
    
    # First, determine if we need to use the multi-chain approach
    # by checking if the protein has missing loops or multiple chains
    has_multiple_chains = self._checkMultipleChains()
    
    # Load the AMBER structure with ParmEd - we'll need this in both cases
    amber_structure = pmd.load_file(self.replica.top, xyz=self.replica.crd)
    
    if has_multiple_chains:
        self.log.info("Detected protein with multiple chains or missing loops. Using chain-aware conversion.")
        # Use the hybrid approach to preserve AMBER parameters while fixing chain handling
        self._convertWithHybridApproach(amber_structure)
    else:
        # Use regular ParmEd conversion for simple proteins
        self.log.info("Using standard conversion for single-chain protein.")
        amber_structure.save(self.replica.grotop, overwrite=True, format='gromacs')
        amber_structure.save(self.replica.gro, overwrite=True, format='gro')

def _checkMultipleChains(self):
    """
    Check if the protein has multiple chains or missing loops.
    
    Returns:
        bool: True if multiple chains or missing loops are detected
    """
    # Try to identify chains from the reference structure
    try:
        from PDB import SolvatedPDB
        import Biskit as bi
        
        # Check if we have a reference PDB file
        ref_pdb = None
        if hasattr(self.replica.system, 'ref') and self.replica.system.ref:
            ref_pdb = self.replica.system.ref.filename
        elif hasattr(self.replica, 'ref') and self.replica.ref:
            ref_pdb = self.replica.ref
        
        if ref_pdb and os.path.exists(ref_pdb):
            # Load the PDB file using Biskit or SolvatedPDB
            solv_pdb = SolvatedPDB(ref_pdb)
            
            # Get chain information
            residue_ids = solv_pdb.chainResIDs()
            
            # If there's more than one chain, return True
            if len(residue_ids) > 1:
                self.log.info(f"Detected {len(residue_ids)} chains in protein")
                return True
            
            # Check for discontinuities in residue numbering that might indicate missing loops
            if len(residue_ids) == 1 and residue_ids[0]:
                sorted_res_ids = sorted(residue_ids[0])
                for i in range(len(sorted_res_ids) - 1):
                    if sorted_res_ids[i+1] - sorted_res_ids[i] > 1:
                        self.log.info(f"Detected gap in residue numbering between {sorted_res_ids[i]} and {sorted_res_ids[i+1]}")
                        return True
        
        # If we can't check, assume there might be missing loops
        return True
        
    except Exception as e:
        self.log.warning(f"Error checking for multiple chains: {str(e)}")
        # If we can't check, assume there might be multiple chains
        return True

def _convertWithHybridApproach(self, amber_structure):
    """
    Convert AMBER to GROMACS using a hybrid approach:
    1. Use ParmEd to preserve all force field parameters
    2. Use a temporary pdb2gmx step to fix chain handling
    3. Merge the chain information from pdb2gmx with the parameters from ParmEd
    
    This ensures we maintain all AMBER parameters while fixing chain topology issues.
    
    Args:
        amber_structure: ParmEd structure loaded from AMBER files
    """
    import parmed as pmd
    import tempfile
    import os
    import subprocess as sub
    
    # Create a temporary directory for the conversion process
    with tempfile.TemporaryDirectory() as tmpdir:
        # First, create GROMACS files directly from AMBER using ParmEd
        # This preserves all force field parameters
        parmed_grotop = os.path.join(tmpdir, "parmed.top")
        parmed_gro = os.path.join(tmpdir, "parmed.gro")
        amber_structure.save(parmed_grotop, overwrite=True, format='gromacs')
        amber_structure.save(parmed_gro, overwrite=True, format='gro')
        
        # Second, save a PDB file for chain processing
        temp_pdb = os.path.join(tmpdir, "amber.pdb")
        amber_structure.save(temp_pdb, overwrite=True)
        
        # Map AMBER force field to GROMACS force field for pdb2gmx
        # This ensures we specify proper force field names for GROMACS tools
        gromacs_force_field, gromacs_water_model = self._mapAmberToGromacsFF()
        
        # Generate a chain-aware topology using pdb2gmx
        # This will create a topology that properly handles chains
        pdb2gmx_grotop = os.path.join(tmpdir, "pdb2gmx.top")
        pdb2gmx_gro = os.path.join(tmpdir, "pdb2gmx.gro")
        
        # Run pdb2gmx with chain handling options
        cmd = f"gmx pdb2gmx -f {temp_pdb} -o {pdb2gmx_gro} -p {pdb2gmx_grotop} " \
              f"-chainsep id_and_ter -merge all -ff {gromacs_force_field} -water {gromacs_water_model} -ignh"
        
        self.log.info(f"Creating chain-aware structure template using pdb2gmx")
        proc = sub.Popen(cmd, shell=True, stdin=sub.PIPE, stdout=sub.PIPE, stderr=sub.PIPE)
        stdout, stderr = proc.communicate()
        
        if proc.returncode != 0:
            self.log.error(f"pdb2gmx conversion failed: {stderr.decode() if stderr else 'Unknown error'}")
            self.log.warning("Falling back to standard ParmEd conversion")
            amber_structure.save(self.replica.grotop, overwrite=True, format='gromacs')
            amber_structure.save(self.replica.gro, overwrite=True, format='gro')
            return
        
        # Now we need to merge the chain information from pdb2gmx with the parameters from ParmEd
        # Three options here:
        # 1. Use the pdb2gmx structure with chain info but copy parameters from ParmEd
        # 2. Use the ParmEd structure with parameters but update moleculetype/chain definitions
        # 3. Generate a hybrid file
        
        # Option 2 is generally simpler - keep ParmEd parameters but fix molecule definitions
        self._mergeChainInfoWithParameters(parmed_grotop, pdb2gmx_grotop, self.replica.grotop)
        
        # For the coordinate file, use the pdb2gmx output as it should have the right chain structure
        import shutil
        shutil.copy(pdb2gmx_gro, self.replica.gro)
        
        self.log.info("Successfully created GROMACS files with correct chain handling while preserving AMBER parameters")

def _mapAmberToGromacsFF(self):
    """
    Map AMBER force field to GROMACS force field names.
    
    Returns:
        tuple: (gromacs_force_field, gromacs_water_model)
    """
    # Default force field mapping if we can't determine from FF
    gromacs_force_field = "amber99sb-ildn"
    gromacs_water_model = "tip3p"
    
    # Try to determine GROMACS force field from AMBER FF
    if hasattr(self.replica, 'FF') and self.replica.FF:
        # Map AMBER FF to GROMACS FF
        amber_ff_map = {
            "leaprc.protein.ff14SB": "amber14sb",
            "leaprc.ff14SB": "amber14sb",
            "leaprc.protein.ff99SB": "amber99sb",
            "leaprc.ff99SB": "amber99sb",
            "leaprc.protein.ff99SBildn": "amber99sb-ildn",
            "leaprc.ff99SBildn": "amber99sb-ildn"
        }
        
        # Map AMBER water model to GROMACS water model
        amber_water_map = {
            "leaprc.water.tip3p": "tip3p",
            "leaprc.water.tip4p": "tip4p",
            "leaprc.water.tip4pew": "tip4p-ew",
            "leaprc.water.tip5p": "tip5p",
            "leaprc.water.spce": "spce"
        }
        
        # Try to find matching force field and water model
        for ff in self.replica.FF:
            if ff in amber_ff_map:
                gromacs_force_field = amber_ff_map[ff]
                self.log.info(f"Mapped AMBER force field {ff} to GROMACS force field {gromacs_force_field}")
            if ff in amber_water_map:
                gromacs_water_model = amber_water_map[ff]
                self.log.info(f"Mapped AMBER water model {ff} to GROMACS water model {gromacs_water_model}")
    
    return gromacs_force_field, gromacs_water_model

def _mergeChainInfoWithParameters(self, parmed_top, pdb2gmx_top, output_top):
    """
    Merge the chain information from pdb2gmx with the parameters from ParmEd.
    
    Args:
        parmed_top: Path to topology file generated by ParmEd
        pdb2gmx_top: Path to topology file generated by pdb2gmx
        output_top: Path to output hybrid topology file
    """
    # Read the topologies
    with open(parmed_top, 'r') as f:
        parmed_content = f.read()
    
    with open(pdb2gmx_top, 'r') as f:
        pdb2gmx_content = f.read()
    
    # Extract key sections from each file
    
    # From ParmEd topology, we want to keep:
    # - All parameter sections (atomtypes, bondtypes, angletypes, dihedraltypes, etc.)
    # - Most of the molecule-specific parameters (atoms, bonds, pairs, angles, dihedrals)
    
    # From pdb2gmx topology, we want:
    # - The moleculetype definition with proper chain organization
    # - The correct system topology section
    # - Any position restraint includes
    
    # The strategy:
    # 1. Use the parmed content as the base
    # 2. Replace the moleculetype and system sections with those from pdb2gmx
    # 3. Ensure position restraint includes are preserved
    
    # Extract moleculetype section from pdb2gmx
    import re
    moleculetype_match = re.search(r'\[ moleculetype \].*?\n(.*?)(?=\n\[)', pdb2gmx_content, re.DOTALL)
    if not moleculetype_match:
        self.log.error("Could not find moleculetype section in pdb2gmx topology")
        # Fall back to using the parmed topology directly
        with open(output_top, 'w') as f:
            f.write(parmed_content)
        return
    
    # Extract system section from pdb2gmx
    system_match = re.search(r'\[ system \].*?\n(.*?)(?=\n\[|$)', pdb2gmx_content, re.DOTALL)
    system_section = system_match.group(0) if system_match else ""
    
    # Look for position restraint includes in pdb2gmx
    posres_includes = re.findall(r'#include ".*?posre.*?"', pdb2gmx_content)
    
    # Replace moleculetype section in parmed content
    new_topology = re.sub(
        r'(\[ moleculetype \].*?)(?=\n\[ atoms \])',
        moleculetype_match.group(0),
        parmed_content,
        flags=re.DOTALL
    )
    
    # Replace system section if found
    if system_section:
        new_topology = re.sub(
            r'\[ system \].*?(?=\n\[|$)',
            system_section,
            new_topology,
            flags=re.DOTALL
        )
    
    # Add position restraint includes if not present
    for include in posres_includes:
        if include not in new_topology:
            # Add after the moleculetype section
            new_topology = new_topology.replace(
                '[ atoms ]',
                f'{include}\n[ atoms ]'
            )
    
    # Write the merged topology
    with open(output_top, 'w') as f:
        f.write(new_topology)
    
    self.log.info("Successfully merged chain information with AMBER parameters")
```

2. Modify the `createRestraints()` method to ensure it works with the merged topology:

```python
def createRestraints(self):
    """
    Add restraints for equilibration with improved handling of the unified
    protein structure from the hybrid approach.
    """
    # For proteins processed with the hybrid approach, we can use the simpler approach
    # as chains were already handled properly during conversion
    self.createPartialGROFromComplex('protein')
    self.createRestraintsITP()
    
    # Process extra residues and ligands as usual
    for extraRes in self.replica.system.extraResList:
        self.createPartialGROFromComplex(extraRes)
        self.createRestraintsITP(section_name=extraRes, group_name=extraRes, 
                               force=1000, itp_out=extraRes+'.itp', ifname=extraRes)
                                
    if (self.replica.system.ligandResname != ''):
        ligRes = self.replica.system.ligandResname
        self.createPartialGROFromComplex(ligRes)
        self.createRestraintsITP(section_name=ligRes, group_name=ligRes, 
                               force=1000, itp_out=ligRes+'.itp', ifname=ligRes)
```

## 5. Advantages of the Hybrid Approach

The hybrid approach offers several key advantages:

1. **Parameter Preservation**: All force field parameters from the original AMBER topology are maintained exactly, ensuring consistent simulations across different MD engines.

2. **Proper Chain Handling**: The approach addresses the root cause of chain handling issues by using pdb2gmx's `-chainsep id_and_ter` option to ensure missing loops don't create separate chains.

3. **Robust and Flexible**: The solution works for both simple proteins and complex multi-chain systems, including those with missing loops.

4. **Future Compatibility**: By leveraging standard GROMACS tools in a way that preserves parameters, this approach is likely to remain compatible with future GROMACS versions.

5. **Minimal Changes to Workflow**: The changes are encapsulated in the file conversion process, requiring minimal changes to the overall workflow.

## 6. Testing Plan

The testing plan for the hybrid approach includes:

1. **Unit Tests**:
   - Test the chain detection mechanism on various protein structures
   - Test the AMBER to GROMACS force field mapping functions
   - Test the topology merging functionality
   - Verify parameter preservation between AMBER and final GROMACS files

2. **Integration Tests**:
   - Validate the complete workflow on a multi-chain protein system
   - Ensure restraints are properly applied during equilibration
   - Verify trajectory stability for systems with missing loops
   - Compare energetics between AMBER and GROMACS simulations

3. **Test Systems**:
   - A simple two-chain protein (e.g., an antibody)
   - A complex multi-domain protein with missing loops
   - A protein with non-sequential residue numbering
   - A protein with ligands and cofactors

4. **Validation Metrics**:
   - Proper restraint application (compare forces with expected values)
   - System stability during equilibration
   - Structural integrity of protein chains
   - Energy minimization convergence
   - Conservation of overall system energy
   - Parameter preservation across conversion

## 7. Potential Challenges

1. **Complex Topology Merging**: The merging of topology sections may encounter challenges with very complex topologies
2. **Custom Residues**: Special handling may be needed for systems with custom residues or non-standard amino acids
3. **GROMACS Version Differences**: Different versions of GROMACS may require adjustments to the pdb2gmx command options
4. **Error Handling**: Robust error handling will be needed to detect and recover from issues during conversion

## 8. Settings Integration

The implementation leverages the existing force field settings in pyMDMix rather than introducing new GROMACS-specific parameters. This ensures consistency in force field selection across different MD engines.

Key aspects of the implementation:

1. **Force Field Mapping**: The code automatically maps AMBER force field specifications from the `FF` parameter to appropriate GROMACS force fields.

2. **Water Model Integration**: Similarly, water model specifications are extracted from the `FF` parameter and mapped to GROMACS water models.

3. **Consistent Settings Management**: By using the existing MDSettings mechanism:
   - Users continue to specify force fields in the familiar way
   - All MD engines use the same force field specification
   - Settings can be overridden at replica creation time or in the user's config file

This approach maintains consistency with the rest of the codebase while providing the flexibility needed for different simulation requirements. The mapping between AMBER and GROMACS force fields ensures that users get the expected behavior regardless of which MD engine is used.

## 9. Conclusion

The proposed development plan addresses the limitation with multi-chain proteins in pyMDMix's GROMACS integration by implementing a hybrid approach that combines the strengths of ParmEd and GROMACS pdb2gmx. 

This solution:
1. Preserves all force field parameters from the original AMBER topology
2. Properly handles proteins with multiple chains or missing loops
3. Maintains compatibility with the existing pyMDMix workflow
4. Provides a robust foundation for future enhancements

By leveraging the existing force field mapping system and implementing a careful merging of topology information, this approach ensures that GROMACS simulations will have the same parameters as their AMBER counterparts while solving the multi-chain handling limitations of the current implementation. 