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

## 3. Proposed Solution

We will enhance the GROMACS integration to properly handle multi-chain proteins by:

1. Improving chain identification and selection logic
2. Creating a more robust group definition mechanism
3. Implementing chain-aware restraint generation
4. Adding validation steps to ensure restraints are properly applied

## 4. Implementation Plan

### 4.1 Step 1: Enhanced Chain Identification

**File to modify**: `pyMDMix/GROMACS.py`

**Changes**:
1. Add a new method `identifyChains()` that will:
   - Parse the structure to identify all protein chains/fragments
   - Create a mapping of chain IDs to residue ranges
   - Store this information in the replica object for later use

```python
def identifyChains(self):
    """
    Identify all protein chains/fragments in the structure and create a mapping
    of chain IDs to residue ranges. This is necessary for proper restraint
    application in multi-chain proteins.
    """
    # Use gmx check or another GROMACS tool to analyze the structure
    cmd = "echo 0 | gmx check -f %s -mno"%(self.replica.gro)
    proc = sub.Popen(cmd, shell=True, stdin=sub.PIPE, stdout=sub.PIPE, stderr=sub.PIPE)
    stdout, stderr = proc.communicate()
    
    # Parse output to identify chains and their boundaries
    # Store chains info in replica object
    self.replica.protein_chains = self._parseChainInfo(stdout, stderr)
    self.log.info(f"Identified {len(self.replica.protein_chains)} protein chains/fragments")
```

**Reason**: This step is crucial for identifying all protein fragments/chains so they can be properly restrained later. The current implementation doesn't explicitly identify and handle multiple chains.

### 4.2 Step 2: Improved Group Creation

**File to modify**: `pyMDMix/GROMACS.py`

**Changes**:
1. Modify the `createGroups()` method to handle multi-chain proteins:
   - Create separate groups for each identified chain
   - Create a unified protein group that includes all chains
   - Ensure the selection syntax accounts for disconnected segments

```python
def createGroups(self):
    # Existing code...
    
    # Enhance to support multi-chain proteins
    if hasattr(self.replica, 'protein_chains') and self.replica.protein_chains:
        # Create individual chain groups
        chain_groups = []
        for i, chain in enumerate(self.replica.protein_chains):
            # Create selection for this chain based on residue ranges
            chain_sel = self._createChainSelection(chain)
            groups += f'"{chain_sel}"\n'
            groups += f'name {lastN+5+i} protein_chain_{i}\n'
            chain_groups.append(f'protein_chain_{i}')
        
        # Create a combined group for all protein chains
        if len(chain_groups) > 1:
            groups += f'{" | ".join(chain_groups)}\n'
            groups += f'name {lastN+5+len(chain_groups)} protein_all_chains\n'
    
    # Continue with existing code...
```

**Reason**: Enhanced group creation ensures that GROMACS correctly identifies all protein fragments. Creating both individual chain groups and a combined group allows for flexible restraint application.

### 4.3 Step 3: Chain-Aware Restraint Generation

**File to modify**: `pyMDMix/GROMACS.py`

**Changes**:
1. Modify the `createRestraints()` method to handle multi-chain proteins:
   - Process each chain separately if multiple chains exist
   - Create combined restraint files as needed

```python
def createRestraints(self):
    """
    Add restraints for equilibration with improved multi-chain support.
    """
    # Handle multi-chain proteins
    if hasattr(self.replica, 'protein_chains') and len(self.replica.protein_chains) > 1:
        self.log.info(f"Creating restraints for multi-chain protein ({len(self.replica.protein_chains)} chains)")
        
        # Create restraints for each chain separately
        for i, chain in enumerate(self.replica.protein_chains):
            chain_name = f"protein_chain_{i}"
            self.createPartialGROFromComplex(chain_name)
            self.createRestraintsITP(section_name="protein", group_name=chain_name, 
                                    force=1000, itp_out=f"{chain_name}.itp", 
                                    ifname=f"POSRES_CHAIN_{i}")
            
        # Create a combined restraint activation for all chains
        self._createMultiChainRestraintActivation()
    else:
        # Original single-chain behavior
        self.createPartialGROFromComplex('protein')
        self.createRestraintsITP()
    
    # Continue with existing code for extra residues and ligands...
```

**Reason**: This modification ensures each chain gets its own restraint file, while maintaining the ability to activate all restraints together. This is critical for systems with missing loops where each segment needs to be properly restrained.

### 4.4 Step 4: Multi-Chain Restraint Topology Integration

**File to modify**: `pyMDMix/GROMACS.py`

**Changes**:
1. Add a new method `_createMultiChainRestraintActivation()` that:
   - Creates a combined activation mechanism for all chain restraints
   - Modifies the topology to include a unified restraint directive

```python
def _createMultiChainRestraintActivation(self):
    """
    Create a unified activation mechanism for all chain restraints in the topology.
    This allows enabling all chain restraints with a single define.
    """
    # Create a unified posres define that includes all chain-specific restraints
    posres_combined = "; Combined position restraints for multi-chain protein\n"
    posres_combined += "#ifdef POSRES\n"
    
    for i in range(len(self.replica.protein_chains)):
        posres_combined += f"#define POSRES_CHAIN_{i}\n"
    
    posres_combined += "#endif\n\n"
    
    # Insert this at the top of the topology file
    with open(self.replica.grotop, 'r') as f:
        toplines = f.read()
    
    # Find insertion point (after initial comments)
    insert_point = toplines.find("[ defaults ]")
    if insert_point == -1:
        insert_point = 0  # Fallback to beginning of file
    
    new_topology = toplines[:insert_point] + posres_combined + toplines[insert_point:]
    
    with open(self.replica.grotop, 'w') as f:
        f.write(new_topology)
    
    # Add the unified restraint identifier to the list
    self.replica.restraint_identifiers.append("POSRES")
```

**Reason**: This creates a unified mechanism to enable/disable all chain restraints simultaneously, which is necessary for proper equilibration protocols. It maintains compatibility with the existing workflow while adding support for multi-chain systems.

### 4.5 Step 5: Helper Methods for Chain Operations

**File to modify**: `pyMDMix/GROMACS.py`

**Changes**:
1. Add helper methods for chain processing:

```python
def _parseChainInfo(self, stdout, stderr):
    """
    Parse GROMACS check output to identify chains and their boundaries.
    
    Returns a list of chain information dictionaries with keys:
    - start_res: First residue number in the chain
    - end_res: Last residue number in the chain
    - chain_id: Chain identifier if available
    """
    # Implementation depends on the exact format of gmx check output
    # This is a placeholder for the parsing logic
    chains = []
    # Parse stdout/stderr to extract chain information
    # ...
    return chains

def _createChainSelection(self, chain):
    """
    Create a GROMACS selection string for a specific chain.
    
    Args:
        chain: Chain information dictionary
        
    Returns:
        Selection string for the chain
    """
    # Create selection based on residue ranges
    return f"r {chain['start_res']}-{chain['end_res']} & protein"
```

**Reason**: These helper methods encapsulate the logic for analyzing and selecting chains, making the code more maintainable and easier to extend in the future.

### 4.6 Step 6: Update MDP Files for Multi-Chain Support

**File to modify**: `pyMDMix/GROMACS.py`

**Changes**:
1. Modify the `writeEqInput()` method to handle multi-chain restraints:

```python
def writeEqInput(self, replica=False):
    # Existing code...
    
    # Handle multi-chain proteins for restraints
    if hasattr(replica, 'protein_chains') and len(replica.protein_chains) > 1:
        # For multi-chain systems, use the unified POSRES define
        if replica.hasRestraints:
            mfield = 'define                   = -DPOSRES'
    else:
        # Original behavior
        if replica.hasRestraints:
            mfield = 'define                   = -DPOSRES'
    
    # Continue with existing code...
```

**Reason**: This ensures the MDP files properly reference the unified restraint activation mechanism for multi-chain proteins.

## 5. Testing Plan

To validate the changes, we need a comprehensive testing approach:

1. **Unit Tests**:
   - Test chain identification on proteins with various numbers of chains
   - Test group creation with multi-chain proteins
   - Test restraint generation for fragmented proteins

2. **Integration Tests**:
   - Validate complete workflow on a multi-chain protein system
   - Ensure restraints are properly applied during equilibration
   - Verify trajectory stability for systems with missing loops

3. **Test Systems**:
   - A simple two-chain protein (e.g., an antibody)
   - A complex multi-domain protein with missing loops
   - A protein with non-sequential residue numbering

4. **Validation Metrics**:
   - Proper restraint application (compare forces with expected values)
   - System stability during equilibration
   - Structural integrity of protein chains

## 6. Implementation Sequence

The implementation should follow this order:

1. Chain identification implementation
2. Helper methods for chain analysis
3. Enhanced group creation
4. Chain-aware restraint generation
5. Multi-chain restraint topology integration
6. MDP file updates
7. Unit testing
8. Integration testing
9. Documentation updates

## 7. Backward Compatibility

The proposed changes maintain backward compatibility by:
1. Preserving the original behavior for single-chain proteins
2. Only activating the enhanced multi-chain handling when multiple chains are detected
3. Using the same restraint application mechanism (conditional compilation)
4. Maintaining the same file structure and naming conventions

## 8. Potential Challenges

1. **Residue numbering**: Non-continuous residue numbering across chains may require special handling
2. **Topology complexity**: Complex topologies with multiple chains may need additional parsing logic
3. **Performance**: Processing many chains could impact performance during setup
4. **GROMACS version differences**: Different versions of GROMACS may have variations in output formats and command syntax

## 9. Conclusion

The proposed development plan addresses the limitation with multi-chain proteins in pyMDMix's GROMACS integration. By enhancing chain identification, group creation, and restraint generation, we can properly support proteins with missing loops that are interpreted as separate chains. The changes maintain compatibility with the existing codebase while adding robust support for complex protein topologies. 