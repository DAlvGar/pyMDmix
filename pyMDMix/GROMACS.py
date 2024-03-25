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

        # Load template inputs
        self.loadConfig()

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


    def getBoxFromCRD(self, crd):
        "Read box size from CRD file bottom line"
        boxline = open(crd, 'r').readlines()[-1]
        return npy.array( boxline.split()[0:3] , dtype='float32')

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
        top = osp.basename(replica.top)
        crd = osp.basename(replica.crd)
        gro = osp.basename(replica.crd)+".gro"
        grotop = osp.basename(replica.top)+".top"
        
        extension = 'nc'

        command = False

        if replica.hasRestraints:
            m = self.getRestraintsIndex()
            mfield = self.restrT.substitute({'force':replica.restrForce,'mask':m})
        else:
            mfield = ''

        # CONVERT FROM AMBER TO GROMACS FORMATS FIRST USING amb2gro_top_gro.py
        
        # # min1
        # $GMX grompp -f min1.mdp -c BAA_complex.gro -p topol.top -o BAA_complex_min1.tpr -n i.ndx
        # $GMX mdrun -s BAA_complex_min1.tpr -deffnm BAA_complex_min1 -nt $NT

        # # min2
        # $GMX grompp -f min2.mdp -c BAA_complex_min1.gro -p topol.top -o BAA_complex_min2.tpr -n i.ndx
        # $GMX mdrun -s BAA_complex_min2.tpr -deffnm BAA_complex_min2 -nt $NT

        if process == 'min1':
            command = 'amb2gro_top_gro.py -p %s -c %s -t %s -g %s \n'%(prevsep+top, prevsep+crd, prevsep+grotop, prevsep+gro)
            command += S.GROMACS_EXE+' grompp -f min1.mdp -c %s -p %s -o min1.tpr -n i.ndx \n'%(prevsep+gro, prevsep+grotop)
            command += S.GROMACS_EXE+' mdrun -s min1.tpr -deffnm min1 -nt %d '%(self.replica.num_threads)
            return command
        
        elif process == 'min2':
            command = S.GROMACS_EXE+' grompp -f min2.mdp -c min1.gro -p %s -o min2.tpr -n i.ndx \n'%(prevsep+grotop)
            command += S.GROMACS_EXE+' mdrun -s min2.tpr -deffnm min2 -nt %d '%(self.replica.num_threads)
            return command

        # EQUILIBRATION STEPS
        
        # # NVT heating posre - 1ns
        # $GMX grompp -f nvt.mdp -c BAA_complex_min2.gro -p topol.top -o BAA_complex_heat.tpr -n i.ndx
        # $GMX mdrun -s BAA_complex_heat.tpr -deffnm BAA_complex_heat -nt $NT

        # # NPT Equil Berendsen no pos res
        # $GMX grompp -f NPT_B.mdp -c BAA_complex_heat.gro -p topol.top -o BAA_complex_equilB.tpr -t BAA_complex_heat.cpt -n i.ndx
        # $GMX mdrun -s BAA_complex_equilB.tpr -deffnm BAA_complex_equilB -nt $NT

        # # NPT Equil Parrinello Rahman no pos res
        # $GMX grompp -f NPT_PR.mdp -c BAA_complex_equilB.gro -p topol.top -o BAA_complex_equilPR.tpr -t BAA_complex_equilB.cpt -n i.ndx
        # $GMX mdrun -s BAA_complex_equilPR.tpr -deffnm BAA_complex_equilPR -nt $NT

        elif process == 'eq':
            if not step: return False

            if step == 1:
                #First step
                # eqfname = replica.eqoutfiletemplate.format(step=step,extension='')
                command = S.GROMACS_EXE+' grompp -f eq1.mdp -c %smin2.gro -p %s -o eq1.tpr -n i.ndx \n'%(prevsep+replica.minfolder+os.sep, prevsep+grotop)
                command += S.GROMACS_EXE+' mdrun -s eq1.tpr -deffnm eq1 -nt %d '%(self.replica.num_threads)

            elif step > 1:
                # eqfname = replica.eqoutfiletemplate.format(step=step,extension='')
                # preveqfname = replica.eqoutfiletemplate.format(step=step-1,extension='')
                command = S.GROMACS_EXE+' grompp -f eq%d.mdp -c eq%d.gro -p %s -o eq%d.tpr -n i.ndx \n'%(step,step-1,prevsep+grotop,step)
                command += S.GROMACS_EXE+' mdrun -s eq%d.tpr -deffnm eq%d -nt %d '%(step,step,self.replica.num_threads)
                # command = S.GROMACS_EXE+' eq%i.py %s %srst %srst '%(step, prevsep+top, preveqfname, eqfname)

            return command
        
        

        elif process == 'md':
            if not step: return False

            mdouttemplate=replica.mdoutfiletemplate.replace('.{extension}','')
            if step == 1:
                fname = mdouttemplate.format(step=1)
                command = S.GROMACS_EXE+' grompp -f md.mdp -c %seq%d.gro -p %s -o md%d.tpr -n i.ndx \n'%( prevsep+replica.eqfolder+os.sep, 3, prevsep+grotop, step)
                command += S.GROMACS_EXE+' mdrun -s md%d.tpr -deffnm md%d -nt %d '%(step,step,self.replica.num_threads)
                # command = S.GROMACS_EXE+' md.py %s %seq5.rst %s.rst %s.nc %s.log'%(prevsep+top, prevsep+replica.eqfolder+os.sep, fname, fname, fname)

                command = command.format(fname=fname)

            elif step > 1:
                prevfname=mdouttemplate.format(step=step-1)
                nextfname=mdouttemplate.format(step=step)
                command = S.GROMACS_EXE+' grompp -f md.mdp -c md%d.gro -p %s -o md%d.tpr -n i.ndx \n'%( step-1, prevsep+grotop, step)
                command += S.GROMACS_EXE+' mdrun -s md%d.tpr -deffnm md%d -nt %d '%(step,step,self.replica.num_threads)
                # command = S.GROMACS_EXE+' md.py %s %s.rst %s.rst %s.nc %s.log'%(prevsep+top, prevfname, nextfname, nextfname, nextfname)
                fname = nextfname
                command = command.format(nextfname=nextfname, prevfname=prevfname)

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
        
        restr = ''
        if replica.hasRestraints:
            if not replica.minimizationAsRef: restr = self.restr
            else: 
                self.log.warn('Use of Minimized structure as restraint reference is still not possible with GROMACS. Will use starting PRMCRD.')
                restr = self.restr
        
        formatdict = {'top':replica.top, 'crd':replica.crd, 'restraints':restr, 
                        'box':self.getBoxFromCRD(replica.crd).max(), 'timestep':int(replica.md_timestep),
                        'freq':replica.trajfrequency}
        # First minimization step 
        # formatdict['minsteps'] = replica.minsteps
        formatdict['minsteps'] = 50000
        out = replica.minfolder+os.sep+'min1.mdp'
        open(out,'w').write(self.minT.substitute(formatdict))
        
        # Smalls teps for conjugate gradient
        formatdict['minsteps'] = 5000
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
        
        restr = ''
        if replica.hasRestraints: restr = self.restr
            
        T.BROWSER.gotoReplica(replica)
        formatdict = {'top':replica.top, 'crd':replica.crd, 'restraints':restr, 
                        'timestep':replica.md_timestep, 'freq':replica.trajfrequency}
        # EQUILIBRATION
        # It comprises 2 steps:
        # - first step (NV) = 500000 steps (1ns) for heating from 100K up to 300K
        # - second step (NPT) = 100000 steps (2ns) of constant pressure constant temp at max temp

        # FIRST STEP
        # Heating up the system from 100 to Final Temperature during 1ns
        formatdict['temp'] = replica.temp
        formatdict['eqinput'] = os.path.join(os.pardir, replica.minfolder, 'min')
        formatdict['eqoutput'] = 'eq1'
        formatdict['first_step'] = 0
        formatdict['final_step'] = replica.namd_heating_steps # 1ns (timestep=0.002ps)
        eq1out = replica.eqfolder+os.sep+'eq1.mdp'
        
        open(eq1out,'w').write(self.eq1T.substitute(formatdict))

        formatdict['eqinput'] = formatdict['eqoutput']
        formatdict['first_step'] = formatdict['final_step']

        # SECOND STEP
        # NPT equilibration for 1ns
        formatdict['eqoutput'] = 'eq2'
        formatdict['final_step'] = formatdict['first_step'] + replica.npt_eq_steps
        eq2out = replica.eqfolder+os.sep+'eq2.mdp'
        open(eq2out,'w').write(self.mdNPT.substitute(formatdict))
        exists = osp.exists(eq1out) and osp.exists(eq2out)
        
        T.BROWSER.goback()
        return exists

    def writeMDInput(self, replica=False):
        """Write production input file for GROMACS in replica.mdfolder
         Args:
            replica     (ReplicaInfo)   Replica instance
        """
        replica = replica or self.replica
        if not replica: raise GROMACSWriterError, "Replica not assigned."
        
        restr = ''
        if replica.hasRestraints: restr = self.restr
            
        T.BROWSER.gotoReplica(replica)

        # PRODUCTION
        # Prepare md configuration files for each trajectory file
        substDict = {'top':replica.top, 'crd':replica.crd, 'restraints':restr,
                        'timestep':replica.md_timestep, 'freq':replica.trajfrequency,
                        'temp':replica.temp}

        # Write md input, 1ns each file and run under NVT conditions
        substDict['nsteps'] = replica.prod_steps # 1ns each file
        outf=replica.mdfolder+os.sep+'md.mdp'
        self.log.debug("Writing: %s"%outf)
        if replica.production_ensemble == 'NPT': prodfile = self.cpmd
        else: prodfile = self.cvmd
        open(outf,'w').write(prodfile.substitute(substDict))
        exists = osp.exists(outf)
        T.BROWSER.goback()
        
        return exists
        
    def writeReplicaInput(self, replica=False):
        replica = replica or self.replica
        if not replica: raise GROMACSWriterError, "Replica not assigned."

        self.log.info("Writing GROMACS simulation input files for replica %s ..."%(replica.name))
        cwd = T.BROWSER.cwd
        T.BROWSER.gotoReplica(replica)

        if not (osp.exists(replica.top) and osp.exists(replica.crd)): # and osp.exists(replica.pdb)):
            raise GROMACSWriterError, "Replica top or crd files not found in current folder: %s, %s"%(replica.top, replica.crd)

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
        if not replica: raise AmberCheckError, "Replica not assigned."
        
        boxextension = boxextension or 'rst'
        
        # Work on step. If not given, fetch last completed production step.
        step = step or replica.lastCompletedProductionStep()
        
        # Fetch rst file and read last line to get box side length and angle
        fname = replica.mdoutfiletemplate.format(step=step, extension=boxextension)
        fname = osp.join(replica.path, replica.mdfolder, fname)
        if not os.path.exists(fname):
            self.log.error("No file found with name %s to fetch box volume in DG0 penalty calculation. Returning no penalty..."%fname)
            return False
        box = map(float, open(fname,'r').readlines()[-1].strip().split())
        vol = box[0]*box[1]*box[2]
        
        if box[3] != 90.0: vol *= 0.77 # orthorombic volume correction
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
        settings = MDSettings(solvent='WAT',mdProgram='GROMACS',restrMode='HA', restrForce=0.1)
        
        self.testdir =  T.tempDir()
        self.r1 = sys+settings
        self.r1.setName('testGROMACS')
        
        T.BROWSER.chdir(self.testdir)
        
        # write replica folder and check methods of GROMACSWriter
        self.r1.createFolder()
        self.r1.createMDInput()
        writer = GROMACSWriter(self.r1)
        
        self.testdir += os.sep+'testGROMACS'
    
    def cleanUp(self):
        T.tryRemove( self.testdir, tree=1 )

if __name__ == "__main__":
    BT.localTest()
