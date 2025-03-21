# GROMACS Integration in pyMDMix

## 1. Introduction

pyMDMix supports the GROMACS molecular dynamics engine as an alternative to the default AMBER engine. This document details the specific workflow for GROMACS system preparation, simulation setup, and analysis integration within the pyMDMix framework.

## 2. GROMACS System Preparation Workflow

The GROMACS workflow in pyMDMix follows these major steps:

1. Converting AMBER format files to GROMACS format
2. Creating index groups for different parts of the system
3. Setting up positional restraints
4. Generating configuration files for minimization, equilibration, and production
5. Creating simulation run commands

### 2.1 System Conversion

The conversion from AMBER to GROMACS formats is handled by the `convertAmberToGromacs()` method in the `GROMACSWriter` class:

```python
def convertAmberToGromacs(self, replica=False):
    import parmed as pmd
    replica = replica or self.replica
    if not replica: raise GROMACSWriterError, "Replica not assigned."
    
    # Load the AMBER prmtop and inpcrd files
    self.log.info("Converting Amber TOP/CRD to GROMACS format for replica %s ..."%(replica.name))
    amber = pmd.load_file(self.replica.top, xyz=self.replica.crd)
    self.replica.grotop = self.replica.top.replace('prmtop','top')
    self.replica.gro = self.replica.crd.replace('prmcrd','gro')

    # Save GROMACS top and gro files, along with a PDB file
    amber.save(self.replica.grotop, overwrite=True, format='gromacs')
    amber.save(self.replica.gro, overwrite=True, format='gro')
```

This method:
- Uses the ParmEd library to load AMBER topology and coordinate files
- Converts them to GROMACS format
- Saves the converted files with appropriate extensions

## 3. Group Creation and Index Files

GROMACS requires index files to define groups of atoms for various purposes like temperature coupling, restraints, etc. The `createGroups()` method handles this:

```python
def createGroups(self):
    # create groups ndx file
    if len(self.replica.system.extraResList): extrares = ' | r '+' '.join([er for er in self.replica.system.extraResList])
    else: extrares = ''
    
    solventbox = self.replica.getSolvent()
    cosolvent = ' '.join([co.name for co in solventbox.residues])
            
    if (self.replica.system.ligandResname != ''):
        ligand_res = '"'+self.replica.system.ligandResname+'"'
    else:
        ligand_res = ''
        
    lastN = self.fetchLastGroupN()
        
    exe = "gmx make_ndx -f %s -o groups.ndx"%(self.replica.gro)
    groups = """
    "protein" %s
    name %d protein_extra
    "Protein-H" %s & !a H*
    name %d protein_extra_noh
    "protein_extra" %s
    name %d solute
    r NA+ Na+ CL- Cl- %s
    name %d solvent
"""%(extrares, lastN+1, extrares, lastN+2, ' | '+ligand_res, lastN+3, cosolvent, lastN+4)
```

This method:
- Defines atom groups for protein, solvent, ions, ligands (if present)
- Creates specific groups for restraints application
- Generates a GROMACS index file (groups.ndx) containing these definitions

## 4. Restraint Setup

One of the most important aspects of the GROMACS workflow is the creation and application of positional restraints, handled by two methods:

### 4.1 Creating Partial GRO Files

First, separate GRO files are created for each component that needs restraints:

```python
def createPartialGROFromComplex(self, group_name):
    """Create an independent gro file for a group. This is needed to create proper constraint 
    ITP file apparently. And we only have one big complex GRO file from Amber"""
    self.log.info("GROMACS: Creating independent gro file %s.gro for restraining"%(group_name))
    cmd = "gmx trjconv -s %s -f %s -n groups.ndx -o %s.gro << EOF\n"%(self.replica.gro, self.replica.gro, group_name)
    cmd += "%s\nEOF"%(group_name)
```

### 4.2 Creating Restraint ITP Files

Then, restraint files are generated for each component:

```python
def createRestraints(self):
    """
    Add restraints for equilibration: protein + extrares + ligand. Each will be in a separate file due to GROMACS limitation.
    """
    self.createPartialGROFromComplex('protein')
    self.createRestraintsITP() # default on protein
    for extraRes in self.replica.system.extraResList:
        self.createPartialGROFromComplex(extraRes)
        self.createRestraintsITP(section_name=extraRes, group_name=extraRes, force=1000, itp_out=extraRes+'.itp', ifname=extraRes)
    if (self.replica.system.ligandResname != ''):
        ligRes = self.replica.system.ligandResname
        self.createPartialGROFromComplex(ligRes)
        self.createRestraintsITP(section_name=ligRes, group_name=ligRes, force=1000, itp_out=ligRes+'.itp', ifname=ligRes)
```

### 4.3 Restraint Application in Topology

The restraint ITP files are included in the topology file with conditional compilation directives:

```python
def createRestraintsITP(self, section_name='system1', group_name='protein', force=1000, itp_out='posre.itp', ifname='POSRES'):
    # ...
    posres = """             
; Include Position restraint file
#ifdef %s
#include "%s"
#endif\n
"""%(ifname, itp_out)
    # ...
    # Insert into topology file
    with open(self.replica.grotop,'r') as top:
        toplines = top.readlines()

    new_lines = []
    inside_moleculetype = False
    inserted_posres = False
    for i, line in enumerate(toplines):
        if '[ moleculetype ]' in line:
            if inserted_posres:
                new_lines.append(posres)
                inserted_posres = False
            inside_moleculetype = True
        elif inside_moleculetype:
            if line.strip().startswith('['):
                inside_moleculetype = False
            elif line.strip().startswith(section_name):
                # Mark that posres needs to be inserted before the next [ moleculetype ] section
                inserted_posres = True
        new_lines.append(line)
```

The restraints are enabled in the MD parameter files by using the `-D` preprocessor flag in GROMACS:

```python
if replica.hasRestraints:        
    mfield = 'define                   = -DPOSRES'
else:
    mfield = ''

# equilibration setp 1 restraints on protein+extrares+ligand
eq_restraints = ' '.join(['-D'+i for i in self.replica.restraint_identifiers])
```

## 5. Configuration Files Generation

pyMDMix generates several configuration files for GROMACS:

### 5.1 Minimization

Two minimization steps are configured:

```python
def writeMinInput(self, replica=False):
    # First minimization step 
    formatdict['minsteps'] = replica.gromacs_min1_steps
    out = replica.minfolder+os.sep+'min1.mdp'
    open(out,'w').write(self.minT.substitute(formatdict))
    
    # Small steps for conjugate gradient
    formatdict['minsteps'] = replica.gromacs_min2_steps
    out = replica.minfolder+os.sep+'min2.mdp'
    open(out,'w').write(self.minT2.substitute(formatdict))
```

### 5.2 Equilibration

Three equilibration steps are configured:

```python
def writeEqInput(self, replica=False):
    # FIRST STEP - Heating with restraints
    formatdict['nsteps'] = replica.gromacs_eq1_steps
    eq1out = replica.eqfolder+os.sep+'eq1.mdp'        
    open(eq1out,'w').write(self.eq1T.substitute(formatdict))

    # SECOND STEP - NPT with Berendsen
    formatdict['nsteps'] = replica.gromacs_eq2_steps
    eq2out = replica.eqfolder+os.sep+'eq2.mdp'
    open(eq2out,'w').write(self.eq2T.substitute(formatdict))

    # THIRD STEP - NPT with Parrinello-Rahman
    formatdict['nsteps'] = replica.gromacs_eq3_steps
    eq3out = replica.eqfolder+os.sep+'eq3.mdp'
    open(eq3out,'w').write(self.eq3T.substitute(formatdict))
```

### 5.3 Production

Production MD configuration:

```python
def writeMDInput(self, replica=False):
    # Write md input, 1ns each file and run under NVT conditions
    substDict['nsteps'] = replica.prod_steps # default is 500K = 2ns @4fs
    outf=replica.mdfolder+os.sep+'md.mdp'
    if replica.production_ensemble == 'NPT': prodfile = self.cpmd
    else: prodfile = self.cvmd
    open(outf,'w').write(prodfile.substitute(substDict))
```

## 6. File Structure and Organization

pyMDMix organizes GROMACS files into a specific directory structure:

### 6.1 Main Files

- `replica.grotop`: GROMACS topology file converted from AMBER
- `replica.gro`: GROMACS coordinate file converted from AMBER
- `groups.ndx`: Index file containing atom group definitions

### 6.2 Restraint Files

- `protein.gro`: Extracted coordinates for protein only
- `posre.itp`: Position restraint file for protein
- `[extraRes].gro`: Extracted coordinates for each extra residue
- `[extraRes].itp`: Position restraint file for each extra residue
- `[ligand].gro`: Extracted coordinates for ligand (if present)
- `[ligand].itp`: Position restraint file for ligand (if present)

### 6.3 Directory Structure

Each replica has the following subdirectories:

- `min/`: Minimization files
  - `min1.mdp`: Configuration for first minimization stage
  - `min2.mdp`: Configuration for second minimization stage
  
- `eq/`: Equilibration files
  - `eq1.mdp`: Configuration for heating stage (NVT with restraints)
  - `eq2.mdp`: Configuration for pressure equilibration (NPT Berendsen)
  - `eq3.mdp`: Configuration for final equilibration (NPT Parrinello-Rahman)
  
- `md/`: Production MD files
  - `md.mdp`: Configuration for production runs

## 7. Simulation Commands

pyMDMix generates shell commands to run the simulations:

```python
def getReplicaCommands(self, replica=None):
    # MINIMIZATION
    outcommands.append('cd %s'%replica.minfolder)
    outcommands.append(self.getCommand('min1'))
    outcommands.append(self.getCommand('min2'))

    # EQUILIBRATION
    outcommands.append('cd %s'%osp.join(os.pardir,replica.eqfolder))
    [outcommands.append(self.getCommand('eq',i)) for i in range(1,4)]

    # PRODUCTION
    outcommands.append('cd %s'%osp.join(os.pardir,replica.mdfolder))
    [outcommands.append(self.getCommand('md',i)) for i in range(1, replica.ntrajfiles+1)]
```

Example command for equilibration:

```python
if step == 1:
    #First step with positional restraints
    command = S.GROMACS_EXE+' grompp -f eq1.mdp -c %smin2.gro -p %s -o eq1.tpr -n ../groups.ndx -r %s \n'%(prevsep+replica.minfolder+os.sep, prevsep+replica.grotop, prevsep+replica.gro)
    command += S.GROMACS_EXE+' mdrun -s eq1.tpr -deffnm eq1 -nt %d -pin auto -nb gpu'%(self.replica.num_threads)
```

## 8. Pre-analysis Trajectory Processing

Before analysis, GROMACS trajectories need special processing:

```python
def preAlign(self, run=True, cmdfile="prealign_trajectory.sh", steps=[]):
    """
    Use gmx trajconv to image and center the trajectory for cpptraj to correctly process it afterwards
    Will act on md production trajectories, replacing the original output by an imaged / centered one 
    """
    if not len(steps): steps = range(1, self.replica.ntrajfiles+1)
    self.replica.go() # Go to replica main folder
    self.log.info("Prealigning GROMACS trajectories")
    
    # Expected extension names in production folder
    exts = self.replica.checkProductionExtension(steps)
    p = self.replica.mdfolder+os.sep
    all_cmds = ""
    for i in steps:
        ext = exts[i]
        n = self.replica.mdoutfiletemplate.format(step=i, extension=ext)
        trajin = p+n # eg md/md1.xtc
        
        if not osp.exists(trajin): 
            raise GROMACSWriterError, "Gromacs trajectory file for step %d not found"%(i)
        
        tprin = trajin.replace(ext, 'tpr')
        trajout=trajin.replace('.'+ext, '_al.'+ext)
        trajtmp=trajin.replace('.'+ext, '_tmp.'+ext)
        cmd = "echo '1 0'|gmx trjconv -s %s -f %s -o %s -pbc nojump -center;\
              echo '1 0'|gmx trjconv -s %s -f %s -o %s -pbc mol -ur compact -center; mv %s %s; rm %s\n"%(tprin, 
                                    trajin, trajtmp, tprin, trajtmp, trajout, trajout, trajin, trajtmp) # 1=center on protein 0=output all system
```

This ensures trajectories are properly imaged and centered before analysis.

## 9. Limitations and Considerations

There are several limitations when using GROMACS with pyMDMix:

1. **Multi-chain proteins**: Proteins with more than one chain might fail in the restraints file preparation due to limitations in how GROMACS handles these restraints.

2. **Advanced restraints**: Applying HA (heavy atom) or position-specific restraints during production simulation is still not fully implemented in the GROMACS workflow.

3. **Ligand handling**: For protein-ligand complexes, the `LIGANDRES` entry must be specified in the system settings section of the configuration file.

4. **Restraint application**: The restraints are applied using the conditional compilation mechanism in GROMACS, which differs from AMBER's approach.

## 10. Configuration Options

To use GROMACS instead of AMBER, add the following to your configuration file:

```
[MDSETTINGS]
MDPROGRAM="GROMACS"
```

Additional GROMACS-specific options include:

- `gromacs_min1_steps`: Steps for first minimization (steepest descent)
- `gromacs_min2_steps`: Steps for second minimization (conjugate gradient)
- `gromacs_eq1_steps`: Steps for heating equilibration
- `gromacs_eq2_steps`: Steps for NPT Berendsen equilibration
- `gromacs_eq3_steps`: Steps for NPT Parrinello-Rahman equilibration
- `production_ensemble`: Can be "NPT" or "NVT"

## 11. Conclusion

The GROMACS integration in pyMDMix provides an alternative MD engine option that follows the same overall workflow as the AMBER implementation. The system is prepared via conversion from AMBER formats, specific GROMACS configurations are generated, and restraints are applied using GROMACS's conditional compilation mechanism. The file structure is organized to maintain consistency with the rest of pyMDMix while accommodating GROMACS-specific requirements. 