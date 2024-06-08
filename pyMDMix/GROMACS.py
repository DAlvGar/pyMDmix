##
## pyMDMix --- http://mdmix.sourceforge.net
## Software for preparation, analysis and quality control
## of solvent mixtures molecular dynamics
##
## This program is free software; you can redistribute it and/or
## modify it under the terms of the GNU General Public License as
## published by the Free Software Foundation; either version 3 of the
## License, or any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
## General Public License for more details.
##
## You find a copy of the GNU General Public License in the file
## license.txt along with this program; if not, write to the Free
## Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
##
## Please cite your use of pyMDMix in published work:
##
##    TOBEPUBLISHED
##

__author__="daniel"
__date__ ="$Feb 10, 2014 11:36:41 AM$"

#import sys
import os
import os.path as osp
import logging
import string
import numpy as npy
import subprocess as sub
#import MDMix.Config as config
#from MDMix.PDB import PDBManager

import tools as T
import settings as S

class GROMACSWriterError(Exception):
    pass

class GROMACSCheckError(Exception):
    pass

class GROMACSWriter(object):
    def __init__(self, replica=False):
        self.log = logging.getLogger("GROMACSWriter")
        self.replica = replica
        self.replica.extension = 'xtc'

        # Load template inputs
        self.loadConfig()        
        
        # Convert inputs and add groups and restraints if not done already
        if not self.replica.gro:
            self.replica.restraint_identifiers = []
            self.convertAmberToGromacs()
            self.createGroups()
            self.createRestraints()

    def loadConfig(self, min=None, min2=None, eq1=None, eq2=None, eq3=None, mdNPT=None, mdNVT=None, restr=None):
        """
        Load standard template files or user given template files for GROMACS MD configuration.
        All template files should contain expected fieldnames. Files are read and loaded for later substitution.

        :arg str min: Filepath for minimization template file.
        :arg str eq1: Filepath for equilibration first step template file.
        :arg str eq2: Filepath for equilibration second step template file.
        :arg str mdNVT: Filepath for production with NVT ensemble template file.
        :arg str mdNPT: Filepath for production with NPT ensemble template file.
        """
        if min: self.minT = string.Template(open(min,'r').read())
        else:   self.minT = string.Template(open(T.templatesRoot('GROMACS_min1_temp.txt'), 'r').read())
        
        if min2: self.minT2 = string.Template(open(min2,'r').read())
        else:   self.minT2 = string.Template(open(T.templatesRoot('GROMACS_min2_temp.txt'), 'r').read())

        if eq1: self.eq1T = string.Template(open(eq1, 'r').read())
        else:   self.eq1T = string.Template(open(T.templatesRoot('GROMACS_eq1_temp.txt'), 'r').read())

        if eq2: self.eq2T = string.Template(open(eq2, 'r').read())
        else:   self.eq2T = string.Template(open(T.templatesRoot('GROMACS_eq2_temp.txt'), 'r').read())

        if eq3: self.eq3T = string.Template(open(eq3, 'r').read())
        else:   self.eq3T = string.Template(open(T.templatesRoot('GROMACS_eq3_temp.txt'), 'r').read())

        if mdNPT:   self.cpmd = string.Template(open(mdNPT, 'r').read())
        else:       self.cpmd = string.Template(open(T.templatesRoot('GROMACS_prod_NPT_temp.txt'), 'r').read())

        if mdNVT:   self.cvmd = string.Template(open(mdNVT, 'r').read())
        else:       self.cvmd = string.Template(open(T.templatesRoot('GROMACS_prod_NVT_temp.txt'), 'r').read())

        # if restr: self.restrT = string.Template(open(restr, 'r').read().rstrip())
        # else: self.restrT = string.Template(open(T.templatesRoot('GROMACS_restr_templ.txt'), 'r').read().rstrip())

    def fetchLastGroupN(self):
        exe = "gmx make_ndx -f %s -o _tmp_.ndx << EOF\nq\nEOF"%(self.replica.gro)
        proc = sub.Popen(exe, shell=True, stdin = sub.PIPE,  stdout=sub.PIPE,  stderr=sub.PIPE)
        exit_code = proc.wait()
        if not exit_code:
            stdout = proc.stdout.read()
            relevant_lines = [line.strip() for line in stdout.split('\n') if len(line.strip().split()) > 2 and (':' in line)]
            group_numbers = [int(line.split()[0]) for line in relevant_lines if line.split()[0].isdigit()]
            if group_numbers:
                return max(group_numbers)
            else:
                return None
        return False
    
    def preAlign(self, run=True, cmdfile="prealign_trajectory.sh", steps=[]):
        """
        Use gmx trajconv to image and center the trajectory for cpptraj to correctly process it afterwards
        Will act on md production trajectories, replacing the original output by an imaged / centered one 
        """
        if not len(steps): steps = range(1, self.replica.ntrajfiles+1)
        self.replica.go() # Go to replica main folder
        self.log.info("Prealigning GROMACS trajectories")
        # steps = range(1, self.replica.ntrajfiles+1)
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
            self.log.debug(cmd)
            all_cmds+=cmd
            if run:
                proc = sub.Popen(cmd, shell=True, stdin = sub.PIPE,  stdout=sub.PIPE,  stderr=sub.PIPE)
                exit_code = proc.wait()
                if exit_code: # Exit different to zero means error
                    self.log.error("Could not prealign GROMACS trajectory "+trajin)
                    raise GROMACSWriterError, "Could not prealign GROMACS trajectory "+trajin
                else:
                    self.log.info("Pre-alignment of GROMACS trajectory done: %s"%trajin)        
                    
        with open(cmdfile,'w') as out:
            out.write(all_cmds)
            
        T.BROWSER.goback()
        return True
    
    def createGroups(self):
        # create groups ndx file
        if len(self.replica.system.extraResList): extrares = ' | r '+' '.join([er for er in self.replica.system.extraResList])
        else: extrares = ''
        
        solventbox = self.replica.getSolvent()
        cosolvent = ' '.join([co.name for co in solventbox.residues])
                
        if (self.replica.system.ligandResname != ''):
            ligand_res = '"'+self.replica.system.ligandResname+'"' # if ligand_residue name is defined, add to protein group for temperature coupling
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
        if ligand_res != '':
            groups+=ligand_res
            groups+="\nname %d ligand\n"%(lastN+5)
        groups += 'q'
        cmd=exe+" << EOF\n"+groups+"\nEOF"
        self.log.debug(cmd)
        proc = sub.Popen(cmd, shell=True, stdin = sub.PIPE,  stdout=sub.PIPE,  stderr=sub.PIPE)
        exit_code = proc.wait()
        if exit_code: # Exit different to zero means error
            self.log.error("Could not generate GROMACS groups")
            raise GROMACSWriterError, "Could not generate GROMACS groups"
        if os.path.exists('groups.ndx'):
            with open('groups.ndx_creation.sh','w') as o:
                o.write(cmd)
            # include groups in topology
            # with open(self.replica.grotop,'r') as top:
            #     toplines = top.read()
            
            # newtop='#include "groups.ndx"\n'+toplines
            
            # with open(self.replica.grotop, 'w') as topout:
            #     topout.write(newtop)
                
            return True
        else:
            return False
        
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

    def createPartialGROFromComplex(self, group_name):
        """Create an independent gro file for a group. This is needed to create proper constraint 
        ITP file apparently. And we only have one big complex GRO file from Amber"""
        self.log.info("GROMACS: Creating independent gro file %s.gro for restraining"%(group_name))
        cmd = "gmx trjconv -s %s -f %s -n groups.ndx -o %s.gro << EOF\n"%(self.replica.gro, self.replica.gro, group_name)
        cmd += "%s\nEOF"%(group_name)
        self.log.debug(cmd)
        proc = sub.Popen(cmd, shell=True, stdin = sub.PIPE,  stdout=sub.PIPE,  stderr=sub.PIPE)
        exit_code = proc.wait()
        if exit_code: # Exit different to zero means error
            self.log.error("Could not generate group GRO file")
            raise GROMACSWriterError, "Could not generate group GRO file"
        return True

    def createRestraintsITP(self, section_name='system1', group_name='protein', force=1000, itp_out='posre.itp', ifname='POSRES'):
        self.log.info("GROMACS: Adding restraints %s"%(itp_out))
        #cmd = "gmx genrestr -f %s -n groups.ndx -o %s -fc %f %f %f << EOF\n"%(self.replica.gro, itp_out, force, force, force)
        #cmd += "%s\nEOF"%(group_name)
        cmd = "echo 0 | gmx genrestr -f %s -o %s -fc %f %f %f"%(group_name+'.gro', itp_out, force, force, force) # 0 to select all content in group gro file since it was already prepared for the specific group
        posres = """             
; Include Position restraint file
#ifdef %s
#include "%s"
#endif\n
"""%(ifname, itp_out)
        self.log.debug(cmd)
        proc = sub.Popen(cmd, shell=True, stdin = sub.PIPE,  stdout=sub.PIPE,  stderr=sub.PIPE)
        exit_code = proc.wait()
        with open('posre.itp_creation.sh','w') as o:
            o.write(cmd)
            
        if exit_code: # Exit different to zero means error
            self.log.error("Could not generate GROMACS restraints file")
            raise GROMACSWriterError, "Could not generate GROMACS restraints file"
        
        if os.path.exists(itp_out):
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
                        
            with open(self.replica.grotop, 'w') as topout:
                topout.write(''.join(new_lines))
                
            self.replica.restraint_identifiers.append(ifname)
            
            return True
        return False    
        #cmd_ha = "gmx genrestr -f %s -n groups.ndx -o ha.itp -fc 1000 1000 1000 -n protein_extra_noh"%(self.replica.gro)
                        
    def getRestraintsIndex(self, replica=False):
        """
        Get the index of atoms to be restrained.
        If replica.restrMask is 'AUTO', calculate mask from solute residue ids.
        
        :args replica: Replica to obtain mask for. If False, use replica loaded at instantiation.
        :type replica: :class:`~Replicas.Replica`
        
        :returns: index of atoms to be restrained or **False** if replica has FREE restrain mode.
        """
        replica = replica or self.replica
        if not replica: raise GROMACSWriterError, "Replica not assigned."

        if replica.restrMode == 'FREE': return False

        if not replica.restrMask or replica.restrMask.upper() == 'AUTO':
            # Obtain mask from residue ids in solute
            if not replica.system:
                raise GROMACSWriterError, "Replica System not set. Can not generate mask."
            pdb = replica.getPDB()
            pdb.setSoluteSolventMask()

            if not pdb:
                raise GROMACSWriterError, "Error creating SolvatedPDB from System in replica %s"%replica.name
            else:
                #TOCHECK
                out = replica.restrMask

        if replica.restrMode == 'BB':
             # Back bone only
             mask_solute_BB = pdb.maskBB() * pdb.soluteMask
             out = npy.where(mask_solute_BB)[0]

        elif replica.restrMode == 'HA':
             # If you want non-hydrogen ids in the protein side
             mask_solute_noH = pdb.maskHeavy() * pdb.soluteMask
             out = npy.where(mask_solute_noH)[0]

        else:
            return

        self.log.debug("Mask index:")
        self.log.debug(out.tolist())
        return out.tolist()


    def getCommand(self, process, step=False, replica=None):
        """
        Return a command string to execute for the given process and number of step.
        It takes into account if it needs restraints and the output file formats.

        ::
            getCommand('min')   # Return minimization execution command for current replica
            getCommand('eq',1)  # Get first step equilibration execution command
            getCommand('md',4)  # Get 4th step production execution command.

        :arg str process: Process for which to return the exe command
        :type process: string that should be: **min** for minimization or **eq** for equilibration or **md** for production

        :returns: string with execution command.
        """
        replica = replica or self.replica
        if not replica: raise GROMACSWriterError, "Replica not assigned."

        prevsep = os.pardir+os.sep        
        # top = osp.basename(replica.top)
        # crd = osp.basename(replica.crd)
        # gro = osp.basename(replica.crd)+".gro"
        # grotop = osp.basename(replica.top)+".top"
        
        # extension = 'nc'

        if replica.hasRestraints: restr = '-r %s'%(prevsep+replica.gro)
        else: restr = ''

        # CONVERT FROM AMBER TO GROMACS FORMATS FIRST USING amb2gro_top_gro.py
        
        # # min1
        # $GMX grompp -f min1.mdp -c BAA_complex.gro -p topol.top -o BAA_complex_min1.tpr -n ../groups.ndx
        # $GMX mdrun -s BAA_complex_min1.tpr -deffnm BAA_complex_min1 -nt $NT

        # # min2
        # $GMX grompp -f min2.mdp -c BAA_complex_min1.gro -p topol.top -o BAA_complex_min2.tpr -n ../groups.ndx
        # $GMX mdrun -s BAA_complex_min2.tpr -deffnm BAA_complex_min2 -nt $NT

        if process == 'min1':
            # command = 'amb2gro_top_gro.py -p %s -c %s -t %s -g %s \n'%(prevsep+top, prevsep+crd, prevsep+replica.grotop, prevsep+replica.gro)
            command = S.GROMACS_EXE+' grompp -f min1.mdp -c %s -p %s -o min1.tpr -n ../groups.ndx \n'%(prevsep+replica.gro, prevsep+replica.grotop)
            command += S.GROMACS_EXE+' mdrun -s min1.tpr -deffnm min1 -nt %d -pin auto -nb gpu'%(self.replica.num_threads)
            return command
        
        elif process == 'min2':
            command = S.GROMACS_EXE+' grompp -f min2.mdp -c min1.gro -p %s -o min2.tpr -n ../groups.ndx \n'%(prevsep+replica.grotop)
            command += S.GROMACS_EXE+' mdrun -s min2.tpr -deffnm min2 -nt %d -pin auto -nb gpu'%(self.replica.num_threads)
            return command

        # EQUILIBRATION STEPS
        
        # # NVT heating posre - 1ns
        # $GMX grompp -f nvt.mdp -c BAA_complex_min2.gro -p topol.top -o BAA_complex_heat.tpr -n ../groups.ndx
        # $GMX mdrun -s BAA_complex_heat.tpr -deffnm BAA_complex_heat -nt $NT

        # # NPT Equil Berendsen no pos res
        # $GMX grompp -f NPT_B.mdp -c BAA_complex_heat.gro -p topol.top -o BAA_complex_equilB.tpr -t BAA_complex_heat.cpt -n ../groups.ndx
        # $GMX mdrun -s BAA_complex_equilB.tpr -deffnm BAA_complex_equilB -nt $NT

        # # NPT Equil Parrinello Rahman no pos res
        # $GMX grompp -f NPT_PR.mdp -c BAA_complex_equilB.gro -p topol.top -o BAA_complex_equilPR.tpr -t BAA_complex_equilB.cpt -n ../groups.ndx
        # $GMX mdrun -s BAA_complex_equilPR.tpr -deffnm BAA_complex_equilPR -nt $NT

        elif process == 'eq':
            if not step: return False

            if step == 1:
                #First step with positional restraints
                command = S.GROMACS_EXE+' grompp -f eq1.mdp -c %smin2.gro -p %s -o eq1.tpr -n ../groups.ndx -r %s \n'%(prevsep+replica.minfolder+os.sep, prevsep+replica.grotop, prevsep+replica.gro)
                command += S.GROMACS_EXE+' mdrun -s eq1.tpr -deffnm eq1 -nt %d -pin auto -nb gpu'%(self.replica.num_threads)
            elif step > 1:
                # Second and third steps without restriants unless requested
                command = S.GROMACS_EXE+' grompp -f eq%d.mdp -c eq%d.gro -p %s -o eq%d.tpr -n ../groups.ndx -t eq%d.cpt %s\n'%(step, step-1, prevsep+replica.grotop, step, step-1, restr)
                command += S.GROMACS_EXE+' mdrun -s eq%d.tpr -deffnm eq%d -nt %d -pin auto -nb gpu'%(step,step,self.replica.num_threads)
            return command
                
        elif process == 'md':
            if not step: return False
            if step == 1:
                command = S.GROMACS_EXE+' grompp -f md.mdp -c %seq%d.gro -p %s -o md%d.tpr -n ../groups.ndx -t %seq%d.cpt %s\n'%( prevsep+replica.eqfolder+os.sep, 3, prevsep+replica.grotop, step, prevsep+replica.eqfolder+os.sep, 3, restr)
                command += S.GROMACS_EXE+' mdrun -s md%d.tpr -deffnm md%d -nt %d -pin auto -nb gpu'%(step,step,self.replica.num_threads)
            elif step > 1:
                command = S.GROMACS_EXE+' grompp -f md.mdp -c md%d.gro -p %s -o md%d.tpr -n ../groups.ndx -t md%d.cpt %s\n'%( step-1, prevsep+replica.grotop, step, step-1, restr)
                command += S.GROMACS_EXE+' mdrun -s md%d.tpr -deffnm md%d -nt %d -pin auto -nb gpu'%(step,step,self.replica.num_threads)
                # command = S.GROMACS_EXE+' md.py %s %s.rst %s.rst %s.nc %s.log'%(prevsep+top, prevfname, nextfname, nextfname, nextfname)
            return command

        else: pass

    def getReplicaCommands(self, replica=None):
        """
        get a string of commands to execute for running the MD for *replica*.
        It will contain the expected file names for input/output files and directory chamnge commands.

        :args replica: Replica to write execution commands for.
        :type replica: :class:`~Replicas.Replica`
        """
        replica = replica or self.replica
        if not replica: raise GROMACSWriterError, "Replica not assigned."

        # Set variables
        outcommands = []

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

        return outcommands

    def writeCommands(self, replica=False, outfile='COMMANDS.sh'):
        "Write list of commands to run the MD into an output file."
        replica = replica or self.replica
        if not replica: raise GROMACSWriterError, "Replica not assigned."
        commands = self.getReplicaCommands(replica)
        open(outfile,'w').write('\n'.join(commands))
        ok = osp.exists(outfile)
        if ok: self.log.debug("Wrote commands file %s"%outfile)
        else: self.log.warn("COMMANDS.sh not writen!")
        return ok

    def writeMinInput(self, replica=False):
        """Write minimization input file for GROMACS in replica.minfolder
         :args replica: Replica instance. If False, use replica assigned in instantiation.
         :type replica: :class:`~Replicas.Replica`
        """
        replica = replica or self.replica
        if not replica: raise GROMACSWriterError, "Replica not assigned."
        
        T.BROWSER.gotoReplica(replica)
        
        # restr = ''
        # if replica.hasRestraints:
        #     if not replica.minimizationAsRef: restr = self.restr
        #     else: 
        #         self.log.warn('Use of Minimized structure as restraint reference is still not possible with GROMACS. Will use starting PRMCRD.')
        #         restr = self.restr
        
        formatdict = {'timestep':int(replica.md_timestep),
                        'freq':replica.trajfrequency}
        # First minimization step 
        # formatdict['minsteps'] = replica.minsteps
        formatdict['minsteps'] = replica.gromacs_min1_steps
        out = replica.minfolder+os.sep+'min1.mdp'
        open(out,'w').write(self.minT.substitute(formatdict))
        
        # Smalls teps for conjugate gradient
        formatdict['minsteps'] = replica.gromacs_min2_steps
        out = replica.minfolder+os.sep+'min2.mdp'
        open(out,'w').write(self.minT2.substitute(formatdict))
        exists = osp.exists(out)
        T.BROWSER.goback()
        
        return exists

    def writeEqInput(self, replica=False):
        """Write equilibration input file for GROMACS in replica.eqfolder
         Args:
            replica     (ReplicaInfo)   Replica instance
        """
        replica = replica or self.replica
        if not replica: raise GROMACSWriterError, "Replica not assigned."
        
        # restr = ''
        # if replica.hasRestraints: restr = self.restr                
        if replica.hasRestraints:        
            mfield = 'define                   = -DPOSRES'
        else:
            mfield = ''

        # equilibration setp 1 restraints on protein+extrares+ligand
        eq_restraints = ' '.join(['-D'+i for i in self.replica.restraint_identifiers])
            
        T.BROWSER.gotoReplica(replica)
        formatdict = {'timestep':'%.3f'%(replica.md_timestep/1000.), 
                      'maskfield':mfield, 'eqresrtaints':eq_restraints,
                      'freq':replica.trajfrequency, 'temp': replica.temp}
        
        # EQUILIBRATION
        # It comprises 3 steps:
        # - first step (NV) = 500000 steps (1ns) for heating from 100K up to 300K
        # - second step (NPT) = 100000 steps (2ns) of constant pressure constant temp at max temp

        # FIRST STEP
        # Heating up the system from 100 to Final Temperature during 1ns WITH RESTRAINTS        
        formatdict['nsteps'] = replica.gromacs_eq1_steps
        eq1out = replica.eqfolder+os.sep+'eq1.mdp'        
        open(eq1out,'w').write(self.eq1T.substitute(formatdict))

        # SECOND STEP
        # NPT Equil Berendsen no pos res
        formatdict['nsteps'] = replica.gromacs_eq2_steps
        eq2out = replica.eqfolder+os.sep+'eq2.mdp'
        open(eq2out,'w').write(self.eq2T.substitute(formatdict))

        # THIRD STEP
        # NPT Equil Parrinello Rahman no pos res
        formatdict['nsteps'] = replica.gromacs_eq3_steps
        eq3out = replica.eqfolder+os.sep+'eq3.mdp'
        open(eq3out,'w').write(self.eq3T.substitute(formatdict))
        exists = osp.exists(eq1out) and osp.exists(eq2out) and osp.exists(eq3out)
        
        T.BROWSER.goback()
        return exists

    def writeMDInput(self, replica=False):
        """Write production input file for GROMACS in replica.mdfolder
         Args:
            replica     (ReplicaInfo)   Replica instance
        """
        replica = replica or self.replica
        if not replica: raise GROMACSWriterError, "Replica not assigned."
        
        if replica.hasRestraints:        
            mfield = 'define                   = -DPOSRES'
        else:
            mfield = ''
            
        T.BROWSER.gotoReplica(replica)

        # PRODUCTION
        # Prepare md configuration files for each trajectory file
        substDict = {'timestep':'%.3f'%(replica.md_timestep/1000.), 
                     'freq':replica.trajfrequency,
                     'maskfield': mfield,
                     'temp':replica.temp}

        # Write md input, 1ns each file and run under NVT conditions
        substDict['nsteps'] = replica.prod_steps # default is 500K = 2ns @4fs
        outf=replica.mdfolder+os.sep+'md.mdp'
        self.log.debug("Writing: %s"%outf)
        if replica.production_ensemble == 'NPT': prodfile = self.cpmd
        else: prodfile = self.cvmd
        open(outf,'w').write(prodfile.substitute(substDict))
        exists = osp.exists(outf)
        T.BROWSER.goback()
        
        return exists
               
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
        #amber.save(options.pdb, overwrite=True, format='pdb')
        
    
    def writeReplicaInput(self, replica=False):
        replica = replica or self.replica
        if not replica: raise GROMACSWriterError, "Replica not assigned."

        self.log.info("Writing GROMACS simulation input files for replica %s ..."%(replica.name))
        cwd = T.BROWSER.cwd
        T.BROWSER.gotoReplica(replica)

        if not (osp.exists(replica.top) and osp.exists(replica.crd)): # and osp.exists(replica.pdb)):
            raise GROMACSWriterError, "Replica top or crd files not found in current folder: %s, %s"%(replica.top, replica.crd)

        # Convert input Amber top and CRD to GROMACS compatible 
        # self.convertAmberToGromacs()

        # Write inputs
        minok = self.writeMinInput(replica)
        eqok = self.writeEqInput(replica)
        mdok = self.writeMDInput(replica)

        if not (minok and eqok and mdok): 
            raise GROMACSWriterError, "MD input not generated for replica %s"%replica.name
        
        self.log.info("MD Input OK")
        T.BROWSER.goHome()
        return True


class GROMACSCheck(object):
    """
    Class to control execution status of an GROMACS simulation process.
    Will check if MD output files are complete or expected trajectory files exist.

    """
    def __init__(self, replica=False, warn=True, **kwargs):
        """
        :args replica: Replica to study. Assign it now or later in each method call.
        :type replica: :class:`~Replicas.Replica`
        :args bool warn: Print warnings when a file is not found or is incomplete.
        """
        self.log = logging.getLogger("GROMACSCheck")
        self.replica = replica
        self.warn = warn

    def checkMinimization(self, replica=False):
        "Check if minimization run correctly"
        replica = replica or self.replica
        if not replica: raise GROMACSCheckError, "Replica not assigned."

        # Move to replica path if not yet there
        T.BROWSER.gotoReplica(replica)
        # Check mimimisation. Loof for 'Maximum number of minimization cycles reached.' string in output
        if not osp.exists(replica.minfolder+os.sep+'min.log'):
            if self.warn: self.log.warn("Minimization output file not found: %smin.out"%(replica.minfolder+os.sep))
            T.BROWSER.goback()
            return False
        minout = open(replica.minfolder+os.sep+'min.log','r').read()
        T.BROWSER.goback()
        return True

    def checkEquilibration(self, replica=False, stepselection=[], outextension='log'):
        """
        Check if equilibration run correctly.

        :arg replica: Replica under study. Default: Loaded replica at instantiation.
        :type replica: :class:`~Replicas.Replica`

        :args list stepselection: Equilibration file step number to check. If False, check all.

        Returns: True or False
        """
        replica = replica or self.replica
        if not replica: raise GROMACSCheckError, "Replica not assigned."
        if not isinstance(stepselection, list): stepselection=[stepselection]

        # Check all equilibration steps (look for 'Total CPU time:')
        selection = stepselection or range(1, 6)
        for i in selection:
            out = self.getEquilibrationOutputFile(i, replica, outextension)
            if not out:
                if self.warn: self.log.warn("Equilibration output file not found for step %i"%i)
                return False
        return True

    def getEquilibrationOutputFile(self, step, replica=False, outextension='log'):
        """
        Return the file content of the equilibration output file for step *step*. Return **False** if file not found.

        :arg int step: Step number to identify equilibration output file. Will use :attr:`Replica.eqoutfiletemplate` to match names.
        :arg replica: Replica under study. Default: Loaded replica at initialization.
        :type replica: :class:`~Replicas.Replica`
        :arg str outextension: Expected file extension for the MD output.

        :return: file content or **False**
        :rtype: str or bool
        """
        replica = replica or self.replica
        if not replica: raise GROMACSCheckError, "Replica not assigned."

        # Move to replica path if not yet there
        T.BROWSER.gotoReplica(replica)
        mdoutfile = replica.eqfolder+os.sep+replica.eqoutfiletemplate.format(step=step,extension=outextension)
        if not osp.exists(mdoutfile): out = False
        else: out = open(mdoutfile, 'r').read()
        T.BROWSER.goback()
        return out


    def getProductionOutputFile(self, step, replica=False, outextension='log'):
        """
        Return the file content of the production output file for step *step*. Return **False** if file not found.

        :arg int step: Step number to identify production output file. Will use :attr:`Replica.mdoutfiletemplate` to match names.
        :arg replica: Replica under study. Default: Loaded replica at initialization.
        :type replica: :class:`~Replicas.Replica`
        :arg str outextension: Expected file extension for the MD output.

        :return: file content or **False**
        :rtype: str or bool
        """
        replica = replica or self.replica
        if not replica: raise GROMACSCheckError, "Replica not assigned."

        # Move to replica path if not yet there
        T.BROWSER.gotoReplica(replica)
        mdoutfile = replica.mdfolder+os.sep+replica.mdoutfiletemplate.format(step=step,extension=outextension)
        if not osp.exists(mdoutfile): out = False
        else: out = open(mdoutfile, 'r').read()
        T.BROWSER.goback()
        return out

    def checkProduction(self, replica=False, stepselection=[], outextension='log'):
        """
        Check if production run correctly.

        :arg replica: Replica under study. Default: Loaded replica at instantiation.
        :type replica: :class:`~Replicas.Replica`

        :args list stepselection: Production file step number to check. If False, check all.

        Returns: True or False
        """
        replica = replica or self.replica
        if not replica: raise GROMACSCheckError, "Replica not assigned."
        if not isinstance(stepselection, list): stepselection=[stepselection]

        selection = stepselection or range(1, replica.ntrajfiles+1)
        for i in selection:
            out = self.getProductionOutputFile(i, replica, outextension)
            if not out:
                if self.warn: self.log.warn("Production output file not found for step %i"%i)
                return False
        return True

    def checkMD(self, replica=False, returnsteps=False):
        "Check in current folder min/ eq/ and md/ for N nanoseconds taken from project info\
         if 'returnsteps', don't evaluate and just return True or False for min, eq and md steps in a dictironary."
        replica = replica or self.replica
        if not replica: raise GROMACSCheckError, "Replica not assigned."

        stepsdone = {}
        stepsdone['min'] = self.checkMinimization(self.replica)
        stepsdone['eq'] = self.checkEquilibration(self.replica)
        stepsdone['md'] = self.checkProduction(self.replica)

        # Return steps?
        if returnsteps: return stepsdone

        # Evaluate to True or False if all is done or not respectively.
        if npy.sum([stepsdone.values()]) == 3:
            self.log.info("Simulation completed for replica %s"%replica.name)
            return True
        else:
            if self.warn: self.log.warn("Checking replica MD failed. Some steps could not pass the check: %s"%stepsdone)
            return False
        
    def getSimVolume(self, replica=False, step=False, boxextension=False):
        """
        Fetch simulation volume information from restart files. 
        
        :arg Replica replica: Replica to study. If false, will take replica loaded in initalization.
        :arg int step: Step to fetch volume for. If False, will identify last completed production step and use that one.
        :arg str boxextension: Extension for the output file containing the restart information. DEFAULT: rst.
        
        :return float Volume: Simulation volume.
        """
        replica = replica or self.replica
        if not replica: raise GROMACSCheckError, "Replica not assigned."
        
        boxextension = boxextension or 'log'
        
        # Work on step. If not given, fetch last completed production step.
        step = step or replica.lastCompletedProductionStep()
        
        # Fetch rst file and read last line to get box side length and angle
        fname = replica.mdoutfiletemplate.format(step=step, extension=boxextension)
        fname = osp.join(replica.path, replica.mdfolder, fname)
        if not os.path.exists(fname):
            self.log.error("No file found with name %s to fetch box volume in DG0 penalty calculation. Returning no penalty..."%fname)
            return False
        
        # Fetch box information
        # in NPT simulations, the log file will contain the average volume in the end of the file
        # EG:
        #
        #            Box-X          Box-Y          Box-Z
        #      8.33814e+00    7.86127e+00    6.80806e+00
        #
        # This is in nanometers, multiply by 10 for angstroms
        # It will be an orthorombic box if prepared with mdmix, so apply also orthorombic volume correction (0.77)
        #
        box_values = []
        in_section = False
        with open(fname,'r') as f:
            for line in f:
                if 'Box-X' in line:
                    in_section=True # go fetch the next one
                elif in_section:
                    [box_values.append(float(v)) for v in line.split()]
                    in_section =False
                    break # stop loop
                else:
                    continue
        
        vol = npy.prod(npy.array(box_values) * 10) * 0.77        
        return vol

    
import Biskit.test as BT

class Test(BT.BiskitTest):
    """Test"""

    def test_GROMACSWriter(self):
        """Create new replica and write MDinput"""
        from MDSettings import MDSettings
        from Systems import SolvatedSystem
        
        top = osp.join(T.testRoot('pep', 'pep.prmtop'))
        crd = osp.join(T.testRoot('pep', 'pep.prmcrd'))
        sys = SolvatedSystem(name='pep',top=top, crd=crd)
        #settings = MDSettings(solvent='WAT',mdProgram='GROMACS',restrMode='HA', restrForce=0.1)
        settings = MDSettings(solvent='ETA',mdProgram='GROMACS',num_threads=16, nanos=200, md_timestep=4, prod_steps=5000000) # 5M * 4fs = 200ns in each file = 1 single production file
        
        self.testdir =  T.tempDir()
        self.r1 = sys+settings # create replica
        self.r1.setName('testGROMACS')
        
        T.BROWSER.chdir(self.testdir)
        
        # write replica folder and check methods of GROMACSWriter
        self.r1.createFolder()
        self.r1.createMDInput()
        writer = GROMACSWriter(self.r1)
        print "\n".join(writer.getReplicaCommands())
        
        self.testdir += os.sep+'testGROMACS'
    
    def cleanUp(self):
        T.tryRemove( self.testdir, tree=1 )

if __name__ == "__main__":
    BT.localTest()
