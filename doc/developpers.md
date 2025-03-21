# pyMDMix Developer Documentation

## 1. Introduction

pyMDMix is a Python package designed for preparation, analysis, and quality control of solvent mixtures molecular dynamics simulations. It provides a complete workflow from system preparation to analysis of results, with support for multiple MD engines including AMBER, GROMACS, and NAMD.

## 2. Architecture Overview

The architecture of pyMDMix follows an object-oriented design with several key components:

### 2.1 Core Components

- **Project**: Central container that manages systems and replicas
- **System/SolvatedSystem**: Represents molecular structures and their solvated states
- **Replica**: Represents individual simulation runs with specific parameters
- **MDSettings**: Configuration for MD simulation parameters
- **Solvent**: Definitions for different solvent types
- **Grid**: Data structure for analysis results (density/energy grids)

### 2.2 MD Engine Interfaces

pyMDMix supports multiple molecular dynamics engines through adapter classes:
- **AMBER**: Interface for AMBER simulation package
- **GROMACS**: Interface for GROMACS simulation package
- **NAMD**: Interface for NAMD simulation package
- **OpenMM**: Interface for OpenMM simulation package

### 2.3 Analysis Components

- **GridData/GridsManager**: Management and analysis of volumetric data
- **Energy**: Energy calculations from grid data
- **Align**: Trajectory alignment tools
- **Trajectory**: Trajectory processing utilities

### 2.4 Command Interface

The package is accessed through a command-line interface with subcommands:
- **Create**: Set up new projects and systems
- **Add**: Add components to existing projects
- **Remove**: Remove components from projects
- **Queue**: Prepare and submit jobs to queue systems
- **Plot**: Visualization of results
- **Analyze**: Analysis of simulation results
- **Tools**: Utility functions

## 3. Workflows

pyMDMix implements several key workflows:

### 3.1 System Preparation Workflow

1. Create a new project (`Create` command)
2. Define system from PDB or other input formats
3. Define MD parameters via settings files
4. Prepare system for simulation (solvation, parameter assignment)

### 3.2 Simulation Workflow

1. Generate input files for MD engines (`createMDInput` method)
2. Submit simulations to computing resources (`Queue` command)
3. Monitor simulation progress
4. Extend simulations if needed

### 3.3 Analysis Workflow

1. Align trajectories (`Align` module)
2. Calculate density/occupancy grids (`GridData` module)
3. Convert density to free energy (`Energy` module)
4. Visualize and analyze hotspots (`Plot` command)

## 4. Key Classes and Relationships

### 4.1 Project Class

The `Project` class is the central container for pyMDMix workflows. It:
- Manages systems and replicas
- Handles project file I/O
- Coordinates simulation preparation and analysis
- Groups replicas for batch operations

### 4.2 Replica Class

Each `Replica` represents a single simulation run with:
- Reference to the system being simulated
- MD parameters
- Folder structure for inputs/outputs
- Methods to generate MD engine inputs
- Analysis functions on resulting trajectories

### 4.3 MD Engine Adapters

Classes like `GROMACSWriter`, `AmberWriter`, etc. provide adapters that:
- Generate engine-specific input files
- Interpret engine-specific output files
- Handle restraints and other engine-specific features
- Provide checking mechanisms for simulation progress

### 4.4 Grid Analysis

The `Grid` and `GridSpace` classes provide:
- Data structures for volumetric data
- Methods for analyzing hotspots
- Boltzmann-weighted averaging
- File I/O for common grid formats (DX, XPLOR)

## 5. Code Organization

### 5.1 Module Structure

- **pyMDMix/__init__.py**: Package initialization and imports
- **pyMDMix/Projects.py**: Project management
- **pyMDMix/Systems.py**: System definition and handling
- **pyMDMix/Replicas.py**: Simulation replica management
- **pyMDMix/MDSettings.py**: MD parameter settings
- **pyMDMix/AMBER.py**, **pyMDMix/GROMACS.py**, etc.: MD engine interfaces
- **pyMDMix/GridData.py**: Grid data structures
- **pyMDMix/GridsManager.py**: Grid management
- **pyMDMix/Energy.py**: Energy calculations
- **pyMDMix/Align.py**: Trajectory alignment
- **pyMDMix/Commands/**: Command-line interface implementation

### 5.2 Data Flow

1. Input data (PDB files, parameter files) → System objects
2. System + MDSettings → Replica objects
3. Replica objects → MD engine input files
4. MD outputs → Trajectory analysis → Grid objects
5. Grid objects → Energy calculations, visualization, hotspot identification

## 6. Use Cases

### 6.1 Protein-Ligand Binding Site Mapping

1. Create a project with a protein structure
2. Define solvent mixtures of interest
3. Set up multiple replicas with different solvent compositions
4. Run MD simulations
5. Calculate density/energy grids
6. Identify favorable binding sites for different probes

### 6.2 Solvent Distribution Analysis

1. Create a project with a protein-ligand complex
2. Solvate with mixed solvents
3. Run simulations to observe solvent distribution
4. Generate density grids for each solvent component
5. Analyze preferential solvation patterns

### 6.3 Fragment-Based Drug Design Support

1. Analyze protein with probe molecules
2. Generate hotspot maps for different functional groups
3. Combine hotspots to guide fragment linking/growing
4. Validate designs with full MD simulations

## 7. Extension Points

pyMDMix can be extended in several ways:

### 7.1 Adding New Solvent Types

1. Define new solvent in the `Solvents.py` module
2. Add residue templates to the data directory
3. Update the solvent database

### 7.2 Supporting New MD Engines

1. Create a new adapter class following the pattern of existing adapters
2. Implement input generation and output parsing
3. Add templates for the new engine

### 7.3 Implementing New Analysis Methods

1. Add new methods to the GridData/GridsManager classes
2. Implement new commands in the Commands directory
3. Update the analysis workflow

## 8. Development Practices

### 8.1 Testing

The codebase includes test functions in most modules that can be run with:
```
python pyMDMix/test.py all
```

Each class typically includes test methods that demonstrate usage.

### 8.2 Logging

pyMDMix uses Python's logging framework extensively:
- Root logger is set up in settings.py
- Each major class has its own logger
- Log levels can be configured via the command-line interface

### 8.3 Error Handling

Error classes follow a hierarchical structure:
- Base error classes for each module (ProjectError, ReplicaError, etc.)
- Specialized error classes for specific issues (BadFile, BadAttribute, etc.)

## 9. Future Directions

Potential areas for improvement and extension:

1. **Python 3 Compatibility**: Update codebase to support modern Python
2. **Additional MD Engines**: Support for more simulation packages
3. **Advanced Analysis Tools**: Implement new analysis techniques
4. **Parallel Processing**: Enhance performance for grid calculations
5. **API Improvements**: More consistent interfaces across modules
6. **Web Interface**: Develop a web-based frontend

## 10. Conclusion

pyMDMix provides a comprehensive framework for molecular dynamics simulations with mixed solvents, focusing on the analysis of molecular interactions. The modular architecture allows for extension and customization, while the command-line interface provides a user-friendly way to access the functionality. 