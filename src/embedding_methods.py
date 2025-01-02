import numpy as np
import copy
#import psi4
import scipy.linalg
import sys
from pyscf import gto, dft, scf, lib, mp, cc, lo, ao2mo
from pyscf.mp.dfmp2_native import DFRMP2
from pyscf.mp.dfump2_native import DFUMP2
from pyscf.tools import molden
#import os

class Embed:
    """ Class with package-independent embedding methods."""

    def __init__(self, keywords):
        """
        Initialize the Embed class.

        Parameters
        ----------
        keywords (dict): dictionary with embedding options.
        """
        self.keywords = keywords
        self.correlation_energy_shell = []
        self.shell_size = 0
        self.outfile = open(keywords['embedding_output'], 'w')
        return None

    @staticmethod
    def dot(A, B):
        """
        (Deprecated) Computes the trace (dot or Hadamard product) 
        of matrices A and B.
        This has now been replaced by a lambda function in 
        embedding_module.py.

        Parameters
        ----------
        A : numpy.array
        B : numpy.array

        Returns
        -------
        The trace (dot product) of A * B

        """
        return np.einsum('ij, ij', A, B)

    def orbital_rotation(self, orbitals, n_active_aos, ao_overlap = None):
        """
        SVD orbitals projected onto active AOs to rotate orbitals.

        If ao_overlap is not provided, C is assumed to be in an
        orthogonal basis.
        
        Parameters
        ----------
        orbitals : numpy.array
            MO coefficient matrix.
        n_active_aos : int
            Number of atomic orbitals in the active atoms.
        ao_overlap : numpy.array (None)
            AO overlap matrix.

        Returns
        -------
        rotation_matrix : numpy.array
            Matrix to rotate orbitals.
        singular_values : numpy.array
            Singular values.
        """
        if ao_overlap is None:
            orthogonal_orbitals = orbitals[:n_active_aos, :]
        else:
            # \bar{C}_{\text{occ}} = S^{1/2} C_{\text{occ}}
            s_half = scipy.linalg.fractional_matrix_power(ao_overlap, 0.5)
            orthogonal_orbitals = (s_half @ orbitals)[:n_active_aos, :]

        # \bar{C}_{\text{occ}}^A = U^A\Sigma^A V^{*A}
        u, s, v = np.linalg.svd(orthogonal_orbitals, full_matrices=True)
        rotation_matrix = v
        singular_values = s
        return rotation_matrix, singular_values

    def orbital_partition(self, sigma, beta_sigma = None):
        """
        Partition the orbital space by SPADE or all AOs in the
        projection basis. Beta variables are only used for open shells.

        Parameters
        ----------
        sigma : numpy.array
            Singular values.
        beta_sigma : numpy.array (None)
            Beta singular values.

        Returns
        -------
        self.n_act_mos : int
            (alpha) number of active MOs.
        self.n_env_mos : int
            (alpha) number of environment MOs.
        self.beta_n_act_mos : int
            Beta number of active MOs.
        self.beta_n_env_mos : int
            Beta number of environment MOs.
        """
        if self.keywords['partition_method'] == 'spade':
            delta_s = [-(sigma[i+1] - sigma[i]) for i in range(len(sigma) - 1)]
            self.n_act_mos = np.argpartition(delta_s, -1)[-1] + 1
            self.n_env_mos = len(sigma) - self.n_act_mos
        else:
            assert isinstance(self.keywords['occupied_projection_basis'], str),\
                '\n Define a projection basis'
            self.n_act_mos = self.n_active_aos
            self.n_env_mos = len(sigma) - self.n_act_mos

        if self.keywords['low_level_reference'] == 'rhf':
            return self.n_act_mos, self.n_env_mos
        else:
            assert beta_sigma is not None, 'Provide beta singular values'
            if self.keywords['partition_method'] == 'spade':
                beta_delta_s = [-(beta_sigma[i+1] - beta_sigma[i]) \
                    for i in range(len(beta_sigma) - 1)]
                self.beta_n_act_mos = np.argpartition(beta_delta_s, -1)[-1] + 1
                self.beta_n_env_mos = len(beta_sigma) - self.beta_n_act_mos
            else:
                assert isinstance(self.keywords['occupied_projection_basis'], str),\
                    '\n Define a projection basis'
                self.beta_n_act_mos = self.beta_n_active_aos
                self.beta_n_env_mos = len(beta_sigma) - self.beta_n_act_mos
            return (self.n_act_mos, self.n_env_mos, self.beta_n_act_mos,
                    self.beta_n_env_mos)

#    def old_PM_localization(self, C_occ, nao_A, n_active_atoms, frag_charge, mo_occ):
#        # Initialize arrays
#        S = self.ao_overlap
#        nao = self._mol.nao
#        if self._mean_field.mo_occ.shape[0] == 2:
#            nocc = list(mo_occ).count(1)
#            nocc_A = int((sum([self._mol.atom_charge(i) for i in range(n_active_atoms)]) + -1*frag_charge))
#        else:
#            nocc = list(mo_occ).count(2)
#            nocc_A = int((sum([self._mol.atom_charge(i) for i in range(n_active_atoms)]) + -1*frag_charge)/2)
#        nocc_B = int(nocc - nocc_A)
#        print(f'{nao = }')
#        print(f'{nocc = }')
#        print(f'{nocc_A = }')
#        print(f'{nocc_B = }')
#        Frgment_LMO_1 = np.zeros((nao, nocc_A))
#        Frgment_LMO_2 = np.zeros((nao, nocc_B))
#
#        # Run localization
#        #loc = lo.PM(self._mol, mo_coeff=C_occ, init_guess=None)
#        #local_C_occ = loc.kernel()
#        loc = lo.PM(self._mol)
#        local_C_occ = loc.kernel(C_occ)
#        np.savetxt('PM_orbitals.txt', local_C_occ)
#
#        # Identify fragment/environment orbitals
#        N_fragments = 2
#        pop = np.zeros((nocc, N_fragments))
#        for i in range(nocc):
#            loc_C = local_C_occ[:,i]
#            dens = np.outer(loc_C,loc_C)
#            PS = np.dot(dens,S)
#
#            pop[i,0] = np.trace(PS[:nao_A,:nao_A])
#            pop[i,1] = np.trace(PS[nao_A:, nao_A:])
#
#        print('pop')
#        print(pop)
#        pop_order_1 = np.argsort(-1*pop[:, 0])
#        pop_order_2 = np.argsort(-1*pop[:, 1])
#        print(f'{pop_order_1 = }')
#        print(f'{pop_order_2 = }')
#
#        orbid_1 = pop_order_1[:nocc_A]
#        orbid_2 = pop_order_2[:nocc_B]
#        #orbid_1 = [9, 1, 0, 18, 5, 11, 10]
#        #orbid_2 = [12, 16,  2, 17,  8,  4,  3,  7, 14, 13,  6, 15]
#
#        print("orbitals assigned to fragment 1:")
#        print(orbid_1)
#        print("orbitals assigned to fragment 2:")
#        print(orbid_2)
#
#        for i in range(nocc_A):
#            Frgment_LMO_1[:,i] = local_C_occ[:,orbid_1[i]]
#        for i in range(nocc_B):
#            Frgment_LMO_2[:,i] = local_C_occ[:,orbid_2[i]]
#
#        return nocc_A, nocc_B, Frgment_LMO_1, Frgment_LMO_2

    def PM_localization(self, C_occ, nao_A, n_active_atoms, mo_occ, nocc_A):
        # Initialize arrays
        S = self.ao_overlap
        nao = self._mol.nao
        if self._mean_field.mo_occ.shape[0] == 2:
            nocc = list(mo_occ).count(1)
        else:
            nocc = list(mo_occ).count(2)
        nocc_B = int(nocc - nocc_A)
        print(f'{nao = }')
        print(f'{nocc = }')
        print(f'{nocc_A = }')
        print(f'{nocc_B = }')
        Frgment_LMO_1 = np.zeros((nao, nocc_A))
        Frgment_LMO_2 = np.zeros((nao, nocc_B))

        # Run localization
        #loc = lo.PM(self._mol, mo_coeff=C_occ)
        #local_C_occ = loc.kernel()
        if self.keywords['partition_method'] == 'pm':
            loc = lo.PM(self._mol, pop_method='mulliken')
            local_C_occ = loc.kernel(C_occ)
            isstable, mo_new = loc.stability_jacobi()
            while not isstable:
                loc.kernel(mo_new)
                isstable, mo_new = loc.stability_jacobi()
            local_C_occ = mo_new
        elif self.keywords['partition_method'] == 'boys':
            loc = lo.Boys(self._mol)
            local_C_occ = loc.kernel(C_occ)

        # Identify fragment/environment orbitals
        N_fragments = 2
        pop = np.zeros((nocc, N_fragments))
        for i in range(nocc):
            loc_C = local_C_occ[:,i]
            dens = np.outer(loc_C,loc_C)
            PS = np.dot(dens,S)

            pop[i,0] = np.trace(PS[:nao_A,:nao_A])
            pop[i,1] = np.trace(PS[nao_A:, nao_A:])

        print('pop')
        print(pop)
        pop_order_1 = np.argsort(-1*pop[:, 0])
        pop_order_2 = np.argsort(-1*pop[:, 1])
        print(f'{pop_order_1 = }')
        print(f'{pop_order_2 = }')

        orbid_1 = pop_order_1[:nocc_A]
        orbid_2 = pop_order_2[:nocc_B]

        print("orbitals assigned to fragment 1:")
        print(orbid_1)
        print("orbitals assigned to fragment 2:")
        print(orbid_2)

        for i in range(nocc_A):
            Frgment_LMO_1[:,i] = local_C_occ[:,orbid_1[i]]
        for i in range(nocc_B):
            Frgment_LMO_2[:,i] = local_C_occ[:,orbid_2[i]]

        return nocc_A, nocc_B, Frgment_LMO_1, Frgment_LMO_2


    def header(self):
        """Prints the header in the output file."""
        self.outfile.write('\n')
        self.outfile.write(' ' + 75*'-' + '\n')
        self.outfile.write('                               PsiEmbed\n\n')
        self.outfile.write('                        Python Stack for Improved\n')
        self.outfile.write('       and Efficient Methods and Benchmarking in'
                            + ' Embedding Development\n\n')
        self.outfile.write('                            Daniel Claudino\n')
        self.outfile.write('                            September  2019\n')
        self.outfile.write(' ' + 75*'-' + '\n')
        self.outfile.write('\n')
        self.outfile.write(' Main references: \n\n')
        self.outfile.write('     Projection-based embedding:\n')
        self.outfile.write('     F.R. Manby, M. Stella, J.D. Goodpaster,'
                            + ' T.F. Miller. III,\n')
        self.outfile.write('     J. Chem. Theory Comput. 2012, 8, 2564.\n\n')

        if self.keywords['partition_method'] == 'spade':
            self.outfile.write('     SPADE partition:\n')
            self.outfile.write('     D. Claudino, N.J. Mayhall,\n')
            self.outfile.write('     J. Chem. Theory Comput. 2019, 15, 1053.\n\n')

        if 'n_cl_shell' in self.keywords.keys():
            self.outfile.write('     Concentric localization (CL):\n')
            self.outfile.write('     D. Claudino, N.J. Mayhall,\n')
            self.outfile.write('     J. Chem. Theory Comput. 2019, 15, 6085.\n\n')

        if self.keywords['package'].lower() == 'psi4':
            self.outfile.write('     Psi4:\n')
            self.outfile.write('     R. M. Parrish, L. A. Burns, D. G. A. Smith'
                + ', A. C. Simmonett, \n')
            self.outfile.write('     A. E. DePrince III, E. G. Hohenstein'
                + ', U. Bozkaya, A. Yu. Sokolov,\n')
            self.outfile.write('     R. Di Remigio, R. M. Richard, J. F. Gonthier'
                + ', A. M. James,\n') 
            self.outfile.write('     H. R. McAlexander, A. Kumar, M. Saitow'
                + ', X. Wang, B. P. Pritchard,\n')
            self.outfile.write('     P. Verma, H. F. Schaefer III'
                + ', K. Patkowski, R. A. King, E. F. Valeev,\n')
            self.outfile.write('     F. A. Evangelista, J. M. Turney,'
                + 'T. D. Crawford, and C. D. Sherrill,\n')
            self.outfile.write('     J. Chem. Theory Comput. 2017, 13, 3185.')

        if self.keywords['package'].lower() == 'pyscf':
            self.outfile.write('     PySCF:\n')
            self.outfile.write('     Q. Sun, T. C. Berkelbach, N. S. Blunt'
                + ', G. H. Booth, S. Guo, Z. Li,\n')
            self.outfile.write('     J. Liu, J. D. McClain, E. R. Sayfutyarova'
                + ', S. Sharma, S. Wouters,\n')
            self.outfile.write('     and G. K.‐L. Chan,\n')
            self.outfile.write('     WIREs Comput. Mol. Sci. 2018, 8, e1340.')
        self.outfile.write('\n\n')
        self.outfile.write(' ' + 75*'-' + '\n')
        return None

    def print_scf(self, e_act, e_env, two_e_cross, e_act_emb, correction):
        """
        Prints mean-field info from before and after embedding.

        Parameters
        ----------
        e_act : float
            Energy of the active subsystem.
        e_env : float
            Energy of the environment subsystem.
        two_e_cross : float
            Intersystem interaction energy.
        e_act_emb : float
            Energy of the embedded active subsystem.
        correction : float
            Correction from the embedded density.
        """
        self.outfile.write('\n\n Energy values in atomic units\n')
        self.outfile.write(' Embedded calculation: '
            + self.keywords['high_level'].upper()
            + '-in-' + self.keywords['low_level'].upper() + '\n\n')
        if self.keywords['partition_method'] == 'spade':
            if 'occupied_projection_basis' not in self.keywords:
                self.outfile.write(' Orbital partition method: SPADE\n')
            else:
                self.outfile.write((' Orbital partition method: SPADE with ',
                    'occupied space projected onto '
                    + self.keywords['occupied_projection_basis'].upper() + '\n'))
        elif self.keywords['partition_method'] == 'pm':
            self.outfile.write(' Orbital partition method: Pipek-Mezey\n')
        elif self.keywords['partition_method'] == 'boys':
            self.outfile.write(' Orbital partition method: Boys\n')
        else:
            self.outfile.write(' Orbital partition method: All AOs in '
                + self.keywords['occupied_projection_basis'].upper()
                + ' from atoms in A\n')

        self.outfile.write('\n')
        if hasattr(self, 'beta_n_act_mos') == False:
            self.outfile.write(' Number of orbitals in active subsystem: %s\n'
                                % self.n_act_mos)
            self.outfile.write(' Number of orbitals in environment: %s\n'
                                % self.n_env_mos)
        else:
            self.outfile.write(' Number of alpha orbitals in active subsystem:'
                                + ' %s\n' % self.n_act_mos)
            self.outfile.write(' Number of beta orbitals in active subsystem:'
                                + ' %s\n' % self.beta_n_act_mos)
            self.outfile.write(' Number of alpha orbitals in environment:'
                                + ' %s\n' % self.n_env_mos)
            self.outfile.write(' Number of beta orbitals in environment:'
                                + ' %s\n' % self.beta_n_env_mos)
        self.outfile.write('\n')
        self.outfile.write(' --- Before embedding --- \n')
        self.outfile.write(' {:<7} {:<6} \t\t = {:>16.10f}\n'.format('('
            + self.keywords['low_level'].upper() +')', 'E[A]', e_act))
        self.outfile.write(' {:<7} {:<6} \t\t = {:>16.10f}\n'.format('('
            + self.keywords['low_level'].upper() +')', 'E[B]', e_env))
        self.outfile.write(' Intersystem interaction G \t = {:>16.10f}\n'.
            format(two_e_cross))
        self.outfile.write(' Nuclear repulsion energy \t = {:>16.10f}\n'.
            format(self.nre))
        self.outfile.write(' {:<7} {:<6} \t\t = {:>16.10f}\n'.format('('
            + self.keywords['low_level'].upper() + ')', 'E[A+B]',
            e_act + e_env + two_e_cross + self.nre))
        self.outfile.write('\n')
        self.outfile.write(' --- After embedding --- \n')
        self.outfile.write(' Embedded SCF E[A] \t\t = {:>16.10f}\n'.
            format(e_act_emb))
        self.outfile.write(' Embedded density correction \t = {:>16.10f}\n'.
            format(correction))
        self.outfile.write(' Embedded HF-in-{:<5} E[A] \t = {:>16.10f}\n'.
            format(self.keywords['low_level'].upper(),
            e_act_emb + e_env + two_e_cross + self.nre + correction))
        self.outfile.write(' <SD_before|SD_after> \t\t = {:>16.10f}\n'.format(
            abs(self.determinant_overlap)))
        self.outfile.write('\n')
        return None

    def print_summary(self, e_mf_emb):
        """
        Prints summary of CL shells.

        Parameters
        ----------
        e_mf_emb : float
            Mean-field embedded energy.
        """
        self.outfile.write('\n Summary of virtual shell energy convergence\n\n')
        self.outfile.write('{:^8} \t {:^8} \t {:^12} \t {:^16}\n'.format(
            'Shell #', '# active', ' Correlation', 'Total'))
        self.outfile.write('{:^8} \t {:^8} \t {:^12} \t {:^16}\n'.format(
            8*'', 'virtuals', 'energy', 'energy'))
        self.outfile.write('{:^8} \t {:^8} \t {:^12} \t {:^16}\n'.format(
            7*'-', 8*'-', 13*'-', 16*'-'))

        for ishell in range(self.n_cl_shell+1):
            self.outfile.write('{:^8d} \t {:^8} \t {:^12.10f} \t {:^12.10f}\n'\
                .format(ishell, self.shell_size*(ishell+1),
                self.correlation_energy_shell[ishell],
                e_mf_emb + self.correlation_energy_shell[ishell]))

        if (ishell == self.max_shell and
            self.keywords['n_cl_shell'] > self.max_shell):
            n_virtuals = self._n_basis_functions - self.n_act_mos
            n_effective_virtuals = (self._n_basis_functions - self.n_act_mos
                                 - self.n_env_mos)
            self.outfile.write('{:^8} \t {:^8} \t {:^12.10f} \t {:^12.10f}\n'.
                format('Eff.', n_effective_virtuals,
                self.correlation_energy_shell[-1],
                e_mf_emb + self.correlation_energy_shell[-1]))
            self.outfile.write('{:^8} \t {:^8} \t {:^12.10f} \t {:^12.10f}\n'.
                format('Full', n_virtuals, self.correlation_energy_shell[-1],
                e_mf_emb + self.correlation_energy_shell[-1]))
        self.outfile.write('\n')
        return None
    
    def print_sigma(self, sigma, ishell):
        """
        Formats the printing of singular values from the CL shells.

        Parameters
        ----------
        sigma : numpy.array or list
            Singular values.
        ishell :int
            CL shell index.
        """
        self.outfile.write('\n{:>10} {:>2d}\n'.format('Shell #', ishell))
        self.outfile.write('  ------------\n')
        self.outfile.write('{:^5} \t {:^14}\n'.format('#','Singular value'))
        for i in range(len(sigma)):
            self.outfile.write('{:^5d} \t {:>12.10f}\n'.format(i, sigma[i]))
        self.outfile.write('\n')
        return None

    def determinant_overlap(self, orbitals, beta_orbitals = None):
        """
        Compute the overlap between determinants formed from the
        provided orbitals and the embedded orbitals

        Parameters
        ----------
        orbitals : numpy.array
            Orbitals to compute the overlap with embedded orbitals.
        beta_orbitals : numpy.array (None)
            Beta orbitals, if running with references other than RHF.
        """
        if self.keywords['high_level_reference'] == 'rhf' and beta_orbitals == None:
            overlap = self.occupied_orbitals.T @ self.ao_overlap @ orbitals
            u, s, vh = np.linalg.svd(overlap)
            self.determinant_overlap = (
                np.linalg.det(u)*np.linalg.det(vh)*np.prod(s))
        else:
            assert beta_orbitals is not None, '\nProvide beta orbitals.'
            alpha_overlap = (self.alpha_occupied_orbitals.T @ self.ao_overlap
                @ beta_orbitals)
            u, s, vh = np.linalg.svd(alpha_overlap)
            self.determinant_overlap = 0.5*(
                np.linalg.det(u)*np.linalg.det(vh)*np.prod(s))
            beta_overlap = (self.beta_occupied_orbitals.T @ self.ao_overlap
                @ beta_orbitals)
            u, s, vh = np.linalg.svd(beta_overlap)
            self.determinant_overlap += 0.5*(
                np.linalg.det(u)*np.linalg.det(vh)*np.prod(s))
        return None

    def count_shells(self):
        """
        Guarantees the correct number of shells are computed.

        Returns
        -------
        max_shell : int
            Maximum number of virtual shells.
        self.n_cl_shell : int
            Number of virtual shells.
        """
        effective_dimension = (self._n_basis_functions - self.n_act_mos
                            - self.n_env_mos)
        self.max_shell = int(effective_dimension/self.shell_size)-1
        if (self.keywords['n_cl_shell']
            > int(effective_dimension/self.shell_size)):
            self.n_cl_shell = self.max_shell
        elif effective_dimension % self.shell_size == 0:
            self.n_cl_shell = self.max_shell - 1
        else:
            self.n_cl_shell = self.keywords['n_cl_shell']
        return self.max_shell, self.n_cl_shell


class PySCFEmbed(Embed):
    """Class with embedding methods using PySCF."""
    
    def run_mean_field(self, v_emb = None):
        """
        Runs mean-field calculation with PySCF.
        If 'level' is not provided, it runs the a calculation at the level
        given by the 'low_level' key in self.keywords. HF otherwise.

        Parameters
        ----------
        v_emb : numpy.array or list of numpy.array (None)
            Embedding potential.
        """
        self._mol = gto.mole.Mole()
        self._mol.verbose = self.keywords['print_level']
        #self._mol.output = self.keywords['driver_output']
        self._mol.atom = self.keywords['geometry']
        self._mol.max_memory = self.keywords['memory']
        self._mol.basis = self.keywords['basis']
        if self.keywords['ecp']:
            self._mol.ecp = {self.keywords['ecp'][0]:self.keywords['ecp'][1]}
        self._mol.spin = self.keywords['multiplicity']
        self._mol.charge = self.keywords['charge']
        if v_emb is None: # low-level/environment calculation
            self._mol.output = self.keywords['driver_output']
            if self.keywords['low_level'] == 'hf':
                if self.keywords['low_level_reference'].lower() == 'rhf':
                    self._mean_field = scf.RHF(self._mol)
                if self.keywords['low_level_reference'].lower() == 'uhf':
                    self._mean_field = scf.UHF(self._mol)
                if self.keywords['low_level_reference'].lower() == 'rohf':
                    self._mean_field = scf.ROHF(self._mol)
                self.e_xc = 0.0
            else:
                if self.keywords['low_level_reference'].lower() == 'rhf':
                    self._mean_field = dft.RKS(self._mol)
                if self.keywords['low_level_reference'].lower() == 'uhf':
                    self._mean_field = dft.UKS(self._mol)
                if self.keywords['low_level_reference'].lower() == 'rohf':
                    self._mean_field = dft.ROKS(self._mol)
            self._mean_field.conv_tol = self.keywords['e_convergence']
            self._mean_field.xc = self.keywords['low_level']
            #CPi turn on soscf
            if self.keywords['scf_converger'] == 'soscf':
                self._mean_field = self._mean_field.newton()
            if self.keywords['checkfile'] is not None:
                self._mean_field.chkfile = self.keywords['checkfile']
                self._mean_field.init_guess = 'chkfile'
            self._mean_field.kernel()
            if self.keywords['do_delta_scf'] == True:
                if self.keywords['lowest_level_molden']:
                    mf = self._mean_field
                    mol = self._mol
                    molden_file = self.keywords['lowest_level_molden']
                    with open('alpha_'+molden_file, 'w') as f1:
                        molden.header(mol, f1)
                        molden.orbital_coeff(mol, f1, mf.mo_coeff[0], ene=mf.mo_energy[0], occ=mf.mo_occ[0])
                    with open('beta_'+molden_file, 'w') as f1:
                        molden.header(mol, f1)
                        molden.orbital_coeff(mol, f1, mf.mo_coeff[1], ene=mf.mo_energy[1], occ=mf.mo_occ[1])
                mo0 = self._mean_field.mo_coeff
                occ0 = self._mean_field.mo_occ
                swap = self.keywords['swap_orbitals']
                occ0[0][swap[0]] = 0
                occ0[0][swap[1]] = 1
                # New SCF caculation 
                b = scf.UKS(self._mol)
                b.xc = self.keywords['low_level']
                #if self.keywords['delta_checkfile'] is not None:
                #    b.chkfile = self.keywords['delta_checkfile']
                #    b.init_guess = 'chkfile'
                # Construct new dnesity matrix with new occpuation pattern
                dm_u = b.make_rdm1(mo0, occ0)
                # Apply mom occupation principle
                b = scf.addons.mom_occ(b, mo0, occ0)
                # Start new SCF with new density matrix
                b.scf(dm_u)
                self._mean_field = b
                #self._mean_field.mo_occ[0][swap[0]] = 1
                #self._mean_field.mo_occ[0][swap[1]] = 0
                #self._mean_field.mo_coeff[0][:,swap[0]] = b.mo_coeff[0][:,swap[1]]
                #self._mean_field.mo_coeff[0][:,swap[1]] = b.mo_coeff[0][:,swap[0]]

            if self.keywords['low_level'] == 'hf':
                self.v_xc_total = 0.0
                self.e_xc_total = 0.0
            else:
                self.v_xc_total = self._mean_field.get_veff()
                self.e_xc_total = self._mean_field.get_veff().exc
        else: # high-level
            if self.keywords['high_level_reference'].lower() == 'rhf':
                self._mean_field = scf.RHF(self._mol)
            if self.keywords['high_level_reference'].lower() == 'uhf':
                self._mean_field = scf.UHF(self._mol)
            if self.keywords['high_level_reference'].lower() == 'rohf':
                self._mean_field = scf.ROHF(self._mol)
            if self.keywords['low_level_reference'].lower() == 'rhf':
                self._mol.nelectron = 2*self.n_act_mos
                # Check whether this is valid when using FNOs
                self._mean_field.get_hcore = lambda *args: v_emb + self.h_core
            if (self.keywords['low_level_reference'].lower() == 'rohf'
                or self.keywords['low_level_reference'].lower() == 'uhf'):
                self._mol.nelectron = self.n_act_mos + self.beta_n_act_mos
                self._mean_field.get_vemb = lambda *args: v_emb
            self._mean_field.conv_tol = self.keywords['e_convergence']
            if self.keywords['frag_checkfile'] is not None:
                self._mean_field.chkfile = self.keywords['frag_checkfile']
                self._mean_field.init_guess = 'chkfile'
            #CPi turn on soscf
            if self.keywords['scf_converger'] == 'soscf':
                self._mean_field = self._mean_field.newton()
            # Construct new density matrices
            if self.keywords['high_level_reference'] == 'rhf':
                self._mean_field.kernel(dm0=self.act_density)
            else:
                self._mean_field.kernel(dm0=(self.alpha_act_density, self.beta_act_density))
            #if self.keywords['analyze_scf']:
            #    self._mean_field.analyze()
            if self.keywords['do_frag_delta_scf'] == True: # Hasn't been tested !!
                if self.keywords['lowest_level_molden']:
                    mf = self._mean_field
                    mol = self._mol
                    molden_file = self.keywords['lowest_level_molden']
                    with open('alpha_'+molden_file, 'w') as f1:
                        molden.header(mol, f1)
                        molden.orbital_coeff(mol, f1, mf.mo_coeff[0], ene=mf.mo_energy[0], occ=mf.mo_occ[0])
                    with open('beta_'+molden_file, 'w') as f1:
                        molden.header(mol, f1)
                        molden.orbital_coeff(mol, f1, mf.mo_coeff[1], ene=mf.mo_energy[1], occ=mf.mo_occ[1])
                mo0 = self._mean_field.mo_coeff
                occ0 = self._mean_field.mo_occ
                swap = self.keywords['swap_orbitals']
                occ0[0][swap[0]] = 0
                occ0[0][swap[1]] = 1
                # New SCF caculation 
                b = scf.UKS(self._mol)
                if self.keywords['low_level_reference'].lower() == 'rhf':
                    b.mol.nelectron = 2*self.n_act_mos
                    # Check whether this is valid when using FNOs
                    b.get_hcore = lambda *args: v_emb + self.h_core
                if (self.keywords['low_level_reference'].lower() == 'rohf'
                    or self.keywords['low_level_reference'].lower() == 'uhf'):
                    b.mol.nelectron = self.n_act_mos + self.beta_n_act_mos
                    b.get_vemb = lambda *args: v_emb
                b.xc = self.keywords['low_level']
                #if self.keywords['delta_checkfile'] is not None:
                #    b.chkfile = self.keywords['delta_checkfile']
                #    b.init_guess = 'chkfile'
                # Construct new dnesity matrix with new occpuation pattern
                dm_u = b.make_rdm1(mo0, occ0)
                # Apply mom occupation principle
                b = scf.addons.mom_occ(b, mo0, occ0)
                # Start new SCF with new density matrix
                b.scf(dm_u)
                self._mean_field = b
                self._mean_field.mo_occ[0][swap[0]] = 1
                self._mean_field.mo_occ[0][swap[1]] = 0
                self._mean_field.mo_coeff[0][:,swap[0]] = b.mo_coeff[0][:,swap[1]]
                self._mean_field.mo_coeff[0][:,swap[1]] = b.mo_coeff[0][:,swap[0]]

        if self.keywords['low_level_reference'] == 'rhf':
            docc = (self._mean_field.mo_occ == 2).sum()
            self.occupied_orbitals = self._mean_field.mo_coeff[:, :docc]
            self.j, self.k = self._mean_field.get_jk() 
            self.v_xc_total = self._mean_field.get_veff() - self.j
        else:
            if (self.keywords['low_level_reference'] == 'uhf' and v_emb is None
                or self.keywords['high_level_reference'] == 'uhf'
                and v_emb is not None):
                n_alpha = (self._mean_field.mo_occ[0] == 1).sum()
                n_beta = (self._mean_field.mo_occ[1] == 1).sum()
                self.alpha_occupied_orbitals = self._mean_field.mo_coeff[
                    0, :, :n_alpha]
                self.beta_occupied_orbitals = self._mean_field.mo_coeff[
                    1, :, :n_beta]
            if (self.keywords['low_level_reference'] == 'rohf' and v_emb is None
                or self.keywords['high_level_reference'] == 'rohf'
                and v_emb is not None):
                n_beta = (self._mean_field.mo_occ == 2).sum()
                n_alpha = n_beta + (self._mean_field.mo_occ == 1).sum()
                self.alpha_occupied_orbitals = self._mean_field.mo_coeff[:, :n_alpha]
                self.beta_occupied_orbitals = self._mean_field.mo_coeff[:, :n_beta]
            j, k = self._mean_field.get_jk() 
            self.alpha_j = j[0] 
            self.beta_j = j[1]
            self.alpha_k = k[0]
            self.beta_k = k[1]
            self.alpha_v_xc_total = self._mean_field.get_veff()[0] - j[0] - j[1]
            self.beta_v_xc_total = self._mean_field.get_veff()[1] - j[0] - j[1]

        self.alpha = 0.0
        self._n_basis_functions = self._mol.nao
        self.nre = self._mol.energy_nuc()
        self.ao_overlap = self._mean_field.get_ovlp(self._mol)
        self.h_core = self._mean_field.get_hcore(self._mol)
        return None

    def print_molden(self, molden_file):
        mf = self._mean_field
        mol = self._mol
        if self.keywords['low_level_reference'].lower() == 'rhf':
            with open(molden_file, 'w') as f1:
                molden.header(mol, f1)
                molden.orbital_coeff(mol, f1, mf.mo_coeff, ene=mf.mo_energy, occ=mf.mo_occ)
        elif self.keywords['low_level_reference'].lower() == 'uhf':
            with open('alpha_'+molden_file, 'w') as f1:
                molden.header(mol, f1)
                molden.orbital_coeff(mol, f1, mf.mo_coeff[0], ene=mf.mo_energy[0], occ=mf.mo_occ[0])
            with open('beta_'+molden_file, 'w') as f1:
                molden.header(mol, f1)
                molden.orbital_coeff(mol, f1, mf.mo_coeff[1], ene=mf.mo_energy[1], occ=mf.mo_occ[1])

    def count_active_aos(self, basis = None):
        """
        Computes the number of AOs from active atoms.

        Parameters
        ----------
        basis : str
            Name of basis set from which to count active AOs.
        
        Returns
        -------
            self.n_active_aos : int
                Number of AOs in the active atoms.
        """
        if basis is None:
            self.n_active_aos = self._mol.aoslice_nr_by_atom()[
                self.keywords['n_active_atoms']-1][3]
        else:
            self._projected_mol = gto.mole.Mole()
            self._projected_mol.spin = self.keywords['multiplicity']
            self._projected_mol.charge = self.keywords['charge']
            self._projected_mol.atom = self.keywords['geometry']
            self._projected_mol.basis = basis 
            self._projected_mf = scf.RHF(self._projected_mol)
            self.n_active_aos = self._projected_mol.aoslice_nr_by_atom()[
                self.keywords['n_active_atoms']-1][3]
        return self.n_active_aos
        
    def basis_projection(self, orbitals, projection_basis):
        """
        Defines a projection of orbitals in one basis onto another.
        
        Parameters
        ----------
        orbitals : numpy.array
            MO coefficients to be projected.
        projection_basis : str
            Name of basis set onto which orbitals are to be projected.

        Returns
        -------
        projected_orbitals : numpy.array
            MO coefficients of orbitals projected onto projection_basis.
        """
        self.projected_overlap = (self._projected_mf.get_ovlp(self._mol)
            [:self.n_active_aos, :self.n_active_aos])
        self.overlap_two_basis = gto.intor_cross('int1e_ovlp_sph', 
            self._mol, self._projected_mol)[:self.n_active_aos, :]
        projected_orbitals = (np.linalg.inv(self.projected_overlap)
                            @ self.overlap_two_basis @ orbitals)
        return projected_orbitals

    def closed_shell_subsystem(self, orbitals):
        """
        Computes the potential matrices J, K, and V and subsystem energies.

        Parameters
        ----------
        orbitals : numpy.array
            MO coefficients of subsystem.

        Returns
        -------
        e : float
            Total energy of subsystem.
        e_xc : float
            DFT Exchange-correlation energy of subsystem.
        j : numpy.array
            Coulomb matrix of subsystem.
        k : numpy.array
            Exchange matrix of subsystem.
        v_xc : numpy.array
            Kohn-Sham potential matrix of subsystem.
        """
        density = 2.0*orbitals @ orbitals.T
        #It seems that PySCF lumps J and K in the J array 
        j = self._mean_field.get_j(dm = density)
        k = np.zeros([self._n_basis_functions, self._n_basis_functions])
        two_e_term =  self._mean_field.get_veff(self._mol, density)
        if self.keywords['low_level'] == 'hf':
            e_xc = 0.0
            v_xc = 0.0 
        else:
            e_xc = two_e_term.exc
            v_xc = two_e_term - j 

        # Energy
        e = self.dot(density, self.h_core + j/2) + e_xc
        return e, e_xc, j, k, v_xc

    def open_shell_subsystem(self, alpha_orbitals, beta_orbitals):
        """
        Computes the potential matrices J, K, and V and subsystem
        energies for open shell cases.

        Parameters
        ----------
        alpha_orbitals : numpy.array
            Alpha MO coefficients.
        beta_orbitals : numpy.array
            Beta MO coefficients.

        Returns
        -------
        e : float
            Total energy of subsystem.
        e_xc : float
            Exchange-correlation energy of subsystem.
        alpha_j : numpy.array
            Alpha Coulomb matrix of subsystem.
        beta_j : numpy.array
            Beta Coulomb matrix of subsystem.
        alpha_k : numpy.array
            Alpha Exchange matrix of subsystem.
        beta_k : numpy.array
            Beta Exchange matrix of subsystem.
        alpha_v_xc : numpy.array
            Alpha Kohn-Sham potential matrix of subsystem.
        beta_v_xc : numpy.array
            Beta Kohn-Sham potential matrix of subsystem.
        """
        alpha_density = alpha_orbitals @ alpha_orbitals.T
        beta_density = beta_orbitals @ beta_orbitals.T

        # J and K
        j = self._mean_field.get_j(dm = [alpha_density, beta_density])
        alpha_j = j[0]
        beta_j = j[1]
        alpha_k = np.zeros([self._n_basis_functions, self._n_basis_functions])
        beta_k = np.zeros([self._n_basis_functions, self._n_basis_functions])
        two_e_term =  self._mean_field.get_veff(self._mol, [alpha_density,
            beta_density])
        if self.keywords['low_level'] == 'hf':
            e_xc = 0.0
            alpha_v_xc = 0.0
            beta_v_xc = 0.0
        else:
            e_xc = two_e_term.exc
            alpha_v_xc = two_e_term[0] - (j[0] + j[1])
            beta_v_xc = two_e_term[1] - (j[0]+j[1])

        # Energy
        e = (self.dot(self.h_core, alpha_density + beta_density)
            + 0.5*(self.dot(alpha_j + beta_j, alpha_density + beta_density))
            + e_xc)

        return e, e_xc, alpha_j, beta_j, alpha_k, beta_k, alpha_v_xc, beta_v_xc
        
    def unrestricted_make_FNO(self, no_cl_shift, thresh=1e-6, pct_occ=None, nvir_act=None, frozen=None):
    
        emb_mf = self._mean_field
        n_frozen_core = self.keywords['n_frozen_core']
        # Run embedded MP2 in canonical basis
        if self.keywords['density_fitting'] == True:
            for i in range(2):
                emb_mf.mo_coeff[i] = np.hstack((emb_mf.mo_coeff[i][:,frozen], 
                                                emb_mf.mo_coeff[i][:,n_frozen_core:no_cl_shift]))
                emb_mf.mo_occ[i] = np.hstack(([1.]*len(frozen), 
                                              emb_mf.mo_occ[i][n_frozen_core:no_cl_shift]))
                emb_mf.mo_energy[i] = np.hstack((emb_mf.mo_energy[i][frozen],
                                                 emb_mf.mo_energy[i][n_frozen_core:no_cl_shift]))
            n_frozen = len(frozen)
            if self.keywords['aux_basis'] is not None:
                emb_dfmp2 = DFUMP2(emb_mf, n_frozen, auxbasis=self.keywords['aux_basis']).run()
            else:
                emb_dfmp2 = DFUMP2(emb_mf, n_frozen).run()
            maskact = ~emb_dfmp2.frozen_mask
            maskocc = [emb_mf.mo_occ[s]>1e-6 for s in [0,1]]
            # dm for a and b spin
            dmab = emb_dfmp2.make_rdm1()
        else:
            emb_mp2 = mp.MP2(emb_mf).set(frozen = frozen).run()
            # dm for a and b spin
            dmab = emb_mp2.make_rdm1(t2=None, with_frozen=False)
    
            maskact = emb_mp2.get_frozen_mask()
            # obtain number of frz occ, act occ, act vir, frz vir
            maskocc = [emb_mp2.mo_occ[s]>1e-6 for s in [0,1]]

        masks = []
        for s in [0,1]:
            masks.append([
                maskocc[s]  & ~maskact[s],  # frz occ
                maskocc[s]  &  maskact[s],  # act occ
                ~maskocc[s] &  maskact[s],  # act vir
                ~maskocc[s] & ~maskact[s],  # frz vir
            ])

        if nvir_act is not None:
            if isinstance(nvir_act, (int, np.integer)):
                nvir_act = [nvir_act]*2
    
        no_frozen = []
        no_coeff = []
        orbital_energies = []
        # loop over a and b spin
        for s,dm in enumerate(dmab):
            if self.keywords['density_fitting'] == True:
                nocc = emb_dfmp2.nocc[s]
                nmo = emb_dfmp2.nmo
            else:
                nocc = emb_mp2.nocc[s]
                nmo = emb_mp2.nmo[s]
            nvir = nmo - nocc
            # get NOONs (n) and NOs eigenvalues (v)
            n,v = np.linalg.eigh(dm[nocc:,nocc:])
            idx = np.argsort(n)[::-1]
            # sort NOONs and eigvs from largest to smallest NOON
            n,v = n[idx], v[:,idx]
            n *= 2  # to match RHF when using same thresh
            print(f'occupation numbers = {n}')
    
            if nvir_act is None:
                if pct_occ is None:
                    # Keep NOs above given threshold
                    nvir_keep = np.count_nonzero(n>thresh)
                else:
                    # Keep given percentage of NOs
                    cumsum = np.cumsum(n/np.sum(n))
                    nvir_keep = np.count_nonzero(
                        [c <= pct_occ or np.isclose(c, pct_occ) for c in cumsum])
            else:
                # Keep fixed number of virtuals
                nvir_keep = min(nvir, nvir_act[s])

            # Get MO energies for different sectors
            moeoccfrz0, moeocc, moevir, moevirfrz0 = [emb_mf.mo_energy[s][m] for m in masks[s]]
            # Get MO coeffs for different sectors
            orboccfrz0, orbocc, orbvir, orbvirfrz0 = [emb_mf.mo_coeff[s][:,m] for m in masks[s]]

            # Collect active virtual MO energies into list
            fvv = np.diag(moevir)
            # natural orbital fock
            fvv_no = np.dot(v.T, np.dot(fvv, v))
            e_vir_canon, v_canon = np.linalg.eigh(fvv_no[:nvir_keep,:nvir_keep])

            # Transform to AO basis?
            orbviract = np.dot(orbvir, np.dot(v[:,:nvir_keep], v_canon))
            orbvirfrz = np.dot(orbvir, v[:,nvir_keep:])

            # Collect different spaces
            # Orbitals
            no_comp = (orboccfrz0, orbocc, orbviract, orbvirfrz, orbvirfrz0)
            no_coeff.append(np.hstack(no_comp))
            # Energies
            mo_e = (moeoccfrz0, moeocc, e_vir_canon, moevir[nvir_keep:], moevirfrz0)
            orbital_energies.append(np.array(np.hstack(mo_e)))

            # Define frozen orbitals
            nocc_loc = np.cumsum([0]+[x.shape[1] for x in no_comp]).astype(int)
            # Create lists with indices of frozen occ and virt orbitals
            no_frozen.append(np.hstack((np.arange(nocc_loc[0], nocc_loc[1]),
                                           np.arange(nocc_loc[3], nocc_loc[5]))).astype(int))

        return no_coeff, orbital_energies, no_frozen

    def restricted_make_FNO(self, no_cl_shift, nvir_act, frozen = None):
        
        emb_mf = self._mean_field
        n_frozen_core = self.keywords['n_frozen_core']
        # Run embedded MP2 in canonical basis
        if self.keywords['density_fitting'] == True:
            emb_mf.mo_coeff = np.hstack((emb_mf.mo_coeff[:,frozen], 
                                            emb_mf.mo_coeff[:,n_frozen_core:no_cl_shift]))
            emb_mf.mo_occ = np.hstack(([2.] * len(frozen), 
                                       emb_mf.mo_occ[n_frozen_core:no_cl_shift]))
            emb_mf.mo_energy = np.hstack((emb_mf.mo_energy[frozen],
                                             emb_mf.mo_energy[n_frozen_core:no_cl_shift]))
            n_frozen = len(frozen)
            if self.keywords['aux_basis'] is not None:
                emb_dfmp2 = DFRMP2(emb_mf, n_frozen, auxbasis=self.keywords['aux_basis']).run()
            else:
                emb_dfmp2 = DFRMP2(emb_mf, n_frozen).run()
            maskact = ~emb_dfmp2.frozen_mask
            maskocc = emb_mf.mo_occ>1e-6
            # dm for a and b spin
            dm = emb_dfmp2.make_rdm1()
        else:
            emb_mp2 = mp.MP2(emb_mf).set(frozen = frozen).run()
            # Construct FNOs (inspired by `mp.make_fno`)
            dm = emb_mp2.make_rdm1(t2=None, with_frozen=False)
            maskact = emb_mp2.get_frozen_mask()
            maskocc = emb_mp2.mo_occ>1e-6

        masks = [maskocc  & ~maskact,    # frz occ
                 maskocc  &  maskact,    # act occ
                 ~maskocc &  maskact,    # act vir
                 ~maskocc & ~maskact]    # frz vir
        
        if self.keywords['density_fitting'] == True:
            nocc = emb_dfmp2.nocc
            nmo = emb_dfmp2.nmo
        else:
            nocc = emb_mp2.nocc
            nmo = emb_mp2.nmo
        #nmo = emb_mp2.nmo
        #nocc = emb_mp2.nocc
        nvir = nmo - nocc
        n,v = np.linalg.eigh(dm[nocc:,nocc:])
        idx = np.argsort(n)[::-1]
        n,v = n[idx], v[:,idx]
        print(f'occupation numbers = {n}')
        
        nvir_keep = min(nvir, nvir_act)

        # mo energies
        moeoccfrz0, moeocc, moevir, moevirfrz0 = [emb_mf.mo_energy[m] for m in masks]
        # mo coefficients
        orboccfrz0, orbocc, orbvir, orbvirfrz0 = [emb_mf.mo_coeff[:,m] for m in masks]
        
        # Canonicalize
        # canonical fock
        fvv = np.diag(moevir)
        # natural orbital fock
        fvv_no = np.dot(v.T, np.dot(fvv, v))
        e_vir_canon, v_canon = np.linalg.eigh(fvv_no[:nvir_keep,:nvir_keep])
        
        # Transform to ?AO? basis
        orbviract = np.dot(orbvir, np.dot(v[:,:nvir_keep], v_canon))
        orbvirfrz = np.dot(orbvir, v[:,nvir_keep:])

        # Collect different spaces
        # Orbitals
        no_comp = (orboccfrz0, orbocc, orbviract, orbvirfrz, orbvirfrz0)
        orbitals = np.hstack(no_comp)
        # Energies
        mo_e = (moeoccfrz0, moeocc, e_vir_canon, moevir[nvir_keep:], moevirfrz0)
        orbital_energies = np.array(np.hstack(mo_e))

        # Define frozen orbitals
        nocc_loc = np.cumsum([0]+[x.shape[1] for x in no_comp]).astype(int)
        no_frozen = np.hstack((np.arange(nocc_loc[0], nocc_loc[1]),
                                  np.arange(nocc_loc[3], nocc_loc[5]))).astype(int)
            
        ## Transform 2e integrals
        # Not needed, this is done in MP2 kernel `_make_eris`! Keeping code here as reference.
        #co = np.asarray(orbocc, order='F')
        #cv = np.asarray(orbviract, order='F')
        #g_mo_sf = ao2mo.general(emb_mf._eri, (co,cv,co,cv), compact=False)

        ## Construct eris array
        #act_mo_energy = np.array(list(moeocc)+list(e_vir_canon))
        #eris = lib.tag_array(g_mo_sf, ovov = g_mo_sf, mo_energy = act_mo_energy)
        
        #return orbitals, orbital_energies, no_frozen, eris
        return orbitals, orbital_energies, no_frozen

    def custom_FNO_MP2(self, emb_mf, n_vir_nos, frozen = None):
        # Construct FNOs and perform FNO-MP2 without using pyscf functions. Was
        # used for debugging. Keeping it here for future reference.
        
        # Run embedded MP2 in canonical basis
        emb_mp2 = mp.MP2(emb_mf).set(frozen = frozen).run()

        # Construct FNOs (copied from `make_fno`)
        t2 = None
        nvir_act = n_vir_nos

        mf = emb_mf
        dm = emb_mp2.make_rdm1(t2=t2, with_frozen=False)
        
        nmo = emb_mp2.nmo
        nocc = emb_mp2.nocc
        nvir = nmo - nocc
        n,v = np.linalg.eigh(dm[nocc:,nocc:])
        idx = np.argsort(n)[::-1]
        n,v = n[idx], v[:,idx]
        print(f'occupation numbers = {n}')
        
        nvir_keep = min(nvir, nvir_act)

        maskact = emb_mp2.get_frozen_mask()
        maskocc = emb_mp2.mo_occ>1e-6
        masks = [maskocc  & ~maskact,    # frz occ
                 maskocc  &  maskact,    # act occ
                 ~maskocc &  maskact,    # act vir
                 ~maskocc & ~maskact]    # frz vir

        # mo energies
        moeoccfrz0, moeocc, moevir, moevirfrz0 = [mf.mo_energy[m] for m in masks]
        # mo coefficients
        orboccfrz0, orbocc, orbvir, orbvirfrz0 = [mf.mo_coeff[:,m] for m in masks]
        
        # Canonicalize
        # canonical fock
        fvv = np.diag(moevir)
        # natural orbital fock
        fvv_no = np.dot(v.T, np.dot(fvv, v))
        e_vir_canon, v_canon = np.linalg.eigh(fvv_no[:nvir_keep,:nvir_keep])
        
        # Transform to ?AO? basis
        orbviract = np.dot(orbvir, np.dot(v[:,:nvir_keep], v_canon))
        orbvirfrz = np.dot(orbvir, v[:,nvir_keep:])

        # Collect different spaces
        no_comp = (orboccfrz0, orbocc, orbviract, orbvirfrz, orbvirfrz0)
        no_coeff = np.hstack(no_comp)

        # Define frozen orbitals
        nocc_loc = np.cumsum([0]+[x.shape[1] for x in no_comp]).astype(int)
        no_frozen = np.hstack((np.arange(nocc_loc[0], nocc_loc[1]),
                                  np.arange(nocc_loc[3], nocc_loc[5]))).astype(int)
            
        ## Calculate MP2 in FNO basis
        fno_mp2 = mp.MP2(emb_mf, mo_coeff=no_coeff, frozen = no_frozen).run()
        fno_e_corr = fno_mp2.e_corr

        # Determine new nocc
        if no_frozen is None:
            nocc = np.count_nonzero(emb_mf.mo_occ > 0)
            assert (nocc > 0)
        elif isinstance(no_frozen, (int, np.integer)):
            nocc = np.count_nonzero(emb_mf.mo_occ > 0) - no_frozen
            assert (nocc > 0)
        elif isinstance(no_frozen[0], (int, np.integer)):
            occ_idx = emb_mf.mo_occ > 0
            occ_idx[list(no_frozen)] = False
            nocc = np.count_nonzero(occ_idx)
            assert (nocc > 0)
        else:
            print('nocc Not working')

        # Determine new nmo
        if no_frozen is None:
            nmo = len(emb_mf.mo_occ)
        elif isinstance(no_frozen, (int, np.integer)):
            nmo = len(emb_mf.mo_occ) - no_frozen
        elif isinstance(no_frozen[0], (int, np.integer)):
            nmo = len(emb_mf.mo_occ) - len(set(no_frozen))
        else:
            print('nmo Not working')

        print('******** Custom FNO-MP2 ********')
        print('nocc = %s, nmo = %s' % (nocc, nmo))
        if frozen is not None:
            print('frozen orbitals %s' % frozen)

        # Get HF energy, which is needed for total MP2 energy.
        print('I am in `get_e_hf`')
        dm   = emb_mf.make_rdm1(no_coeff, emb_mf.mo_occ)
        vhf  = emb_mf.get_veff(emb_mf.mol, dm)
        e_hf = emb_mf.energy_tot(dm=dm, vhf=vhf)
        print(f'{e_hf = }')
        print(f'{emb_mf.e_tot = }')
        print(f'{emb_mf.mo_energy = }')

        no_comp = (orbocc, orbviract)
        act_no_coeff = np.hstack(no_comp)

        print(f'I am rebuilding the fock matrix')
        dm     = emb_mf.make_rdm1(no_coeff, emb_mf.mo_occ)
        vhf    = emb_mf.get_veff(emb_mf.mol, dm)
        fockao = emb_mf.get_fock(vhf=vhf, dm=dm)
        fock = act_no_coeff.conj().T.dot(fockao).dot(act_no_coeff)
        print(f'{fock[nocc] = }')
        mo_energy = fock.diagonal().real

        print("Building fock matrix Fabijan's way")
        #active_mo_energy = []
        #for e,i in enumerate(emb_mf.mo_energy):
        #    if e not in no_frozen:
        #        active_mo_energy.append(i)
        active_mo_energy = np.hstack((moeocc, moevir))

        C_occ = np.identity(nocc)
        # nmo = number of active orbitals
        # nocc + nvir = including all NOs
        C_tot = np.zeros((nocc+nvir, nmo))
        C_tot[:nocc, :nocc] = C_occ
        # v = non-canonicalized NOs
        C_tot[nocc:, nocc:] = v[:,:nvir_keep]
        print(f'{v[:,:nvir_keep].shape = }')

        
        f_mo_sf = np.diag(active_mo_energy)
        f_mo_FNO = np.dot(C_tot.T, np.dot(f_mo_sf, C_tot))
        #e_vir_canon, NO_canon = np.linalg.eigh(f_mo_FNO[nocc:,nocc:])
        print('e_vir_canon = ', e_vir_canon)
        print(f'{C_tot.shape = }')

        print(f'{act_no_coeff.shape = }')
        print(f'{C_tot.shape = }')

        # Construct g_mo_sf!
        print(f'{emb_mp2.mo_coeff.shape = }')
        not_frozen = []
        for i in range(emb_mf.mo_coeff.shape[1]):
            if i not in frozen:
                not_frozen.append(i)
        co = np.asarray(orbocc, order='F')
        #C_ao_canon_fno = np.dot(emb_mf.mo_coeff[:,not_frozen][:,nocc:], np.dot(v[:,:nvir_keep], NO_canon))
        #cv = np.asarray(C_ao_canon_fno, order='F')
        cv = np.asarray(orbviract, order='F')
        print(f'{emb_mf._eri.shape = }')
        print(f'{emb_mf._eri.size = }')
        print(f'{co.shape[0]**4 = }')

        print(f'{ao2mo.restore(1, emb_mf._eri, co.shape[0]).shape = }')
        print(f'{ao2mo.restore(1, emb_mf._eri, co.shape[0]).size = }')
        #emb_mf._eri = ao2mo.restore(1, emb_mf._eri, co.shape[0])

        g_mo_sf = ao2mo.general(emb_mf._eri, (co,cv,co,cv), compact=False)
        print(f'{g_mo_sf.shape = }')

        eris = lib.tag_array(g_mo_sf, ovov = g_mo_sf, mo_energy = list(moeocc)+list(e_vir_canon))
        
        print('Second kernel for calculating MP2 energy.')
        print(f'{mo_energy.shape = }')
        with_t2 = True
        mo_energy = np.array(eris.mo_energy)
        print(f'{mo_energy.shape = }')
    
        nocc = nocc
        nvir = nmo - nocc
        eia = mo_energy[:nocc,None] - mo_energy[None,nocc:]
    
        if with_t2:
            t2 = np.empty((nocc,nocc,nvir,nvir), dtype=eris.ovov.dtype)
        else:
            t2 = None
    
        emp2_ss = emp2_os = 0
        for i in range(nocc):
            if isinstance(eris.ovov, np.ndarray) and eris.ovov.ndim == 4:
                # When mf._eri is a custom integrals with the shape (n,n,n,n), the
                # ovov integrals might be in a 4-index tensor.
                gi = eris.ovov[i]
            else:
                gi = np.asarray(eris.ovov[i*nvir:(i+1)*nvir])
    
            gi = gi.reshape(nvir,nocc,nvir).transpose(1,0,2)
            t2i = gi.conj()/lib.direct_sum('jb+a->jba', eia, eia[i])
            edi = np.einsum('jab,jab', t2i, gi) * 2
            exi = -np.einsum('jab,jba', t2i, gi)
            emp2_ss += edi*0.5 + exi
            emp2_os += edi*0.5
            if with_t2:
                t2[i] = t2i
    
        emp2_ss = emp2_ss.real
        emp2_os = emp2_os.real
        emp2 = lib.tag_array(emp2_ss+emp2_os, e_corr_ss=emp2_ss, e_corr_os=emp2_os)

        print(f'{float(emp2) = }')
        print(f'{float(emp2.e_corr_ss) = }')
        print(f'{float(emp2.e_corr_os) = }')

        fno_e_corr = float(emp2)
        ##### PySCF MP2 continues below

        #mo_energy = None
        #if emb_mf.converged:
        #    # init_amps redirects to kernel in beginning of file.
        #    e_corr, t2 = emb_mf.init_amps(mo_energy, no_coeff, eris)
        #else:
        #    converged, e_corr, t2 = _iterative_kernel(emb_mf, eris)

        #e_corr_ss = getattr(self.e_corr, 'e_corr_ss', 0)
        #e_corr_os = getattr(self.e_corr, 'e_corr_os', 0)
        #fno_e_corr = float(self.e_corr)

        #self._finalize()
        return fno_e_corr

    def correlation_energy(self, span_orbitals = None, kernel_orbitals = None,
        span_orbital_energies = None, kernel_orbital_energies = None):
        """
        Computes the correlation energy for the current set of active
        virtual orbitals.
        
        Parameters
        ----------
        span_orbitals : numpy.array
            Orbitals transformed by the span of the previous shell.
        kernel_orbitals : numpy.array
            Orbitals transformed by the kernel of the previous shell.
        span_orbital_energies : numpy.array
            Orbitals energies of the span orbitals.
        kernel_orbital_energies : numpy.array
            Orbitals energies of the kernel orbitals.

        Returns
        -------
        correlation_energy : float
            Correlation energy of the span_orbitals.
        """

        emb_mf = self._mean_field
        n_vir_nos = self.keywords['n_vir_nos']
        shift = self._n_basis_functions - self.n_env_mos
        if self.keywords['partition_method'] == 'pm' or self.keywords['partition_method'] == 'boys':
            no_cl_shift = self._n_basis_functions - self.n_env_mos 
        else:
            no_cl_shift = self._n_basis_functions - self.n_all_env_mos 
        if span_orbitals is None:
            # If not using CL orbitals, just freeze the level-shifted MOs
            if self.keywords['n_frozen_core'] != 0:
                frozen_orbitals_env = [i for i in range(no_cl_shift, self._n_basis_functions)]
                frozen_core = [i for i in range(self.keywords['n_frozen_core'])]
                frozen_orbitals = frozen_core + frozen_orbitals_env
            else:
                frozen_orbitals = [i for i in range(no_cl_shift, self._n_basis_functions)]
            # Construct FNOs
            if self.keywords['n_vir_nos'] is not None and not self.keywords['high_level'] == 'fno-mp2':
                if self.keywords['high_level_reference'] == 'rhf':
                    orbitals, orbital_energies, frozen_orbitals = self.restricted_make_FNO(
                                                                  no_cl_shift,
                                                                  n_vir_nos,
                                                                  frozen = frozen_orbitals)
                elif self.keywords['high_level_reference'] == 'uhf':
                    orbitals, orbital_energies, frozen_orbitals = self.unrestricted_make_FNO(
                                                                  no_cl_shift,
                                                                  nvir_act=n_vir_nos, 
                                                                  frozen = frozen_orbitals)
                else:
                    print('ROHF not implemented')
                self._mean_field.mo_energy = orbital_energies
                self._mean_field.mo_coeff = orbitals
        else:
            # Preparing orbitals and energies for CL shell
            effective_orbitals = np.hstack((span_orbitals, kernel_orbitals))
            orbital_energies = np.concatenate((span_orbital_energies,
                kernel_orbital_energies))
            frozen_orbitals = [i for i in range(self.n_act_mos
                + span_orbitals.shape[1], self._n_basis_functions)]
            orbitals = np.hstack((self.occupied_orbitals,
                effective_orbitals, self._mean_field.mo_coeff[:, shift:]))
            orbital_energies = (
                np.concatenate((self._mean_field.mo_energy[:self.n_act_mos],
                orbital_energies, self._mean_field.mo_energy[shift:])))
            # Replace orbitals in the mean_field obj by the CL orbitals
            # and compute correlation energy
            self._mean_field.mo_energy = orbital_energies
            self._mean_field.mo_coeff = orbitals

        if self.keywords['high_level'].lower() == 'mp2':
            #if self.keywords['n_vir_nos'] is not None:
            #    embedded_wf = mp.MP2(self._mean_field, frozen = frozen_orbitals)
            #    embedded_wf.kernel()
            #else:
            #if self.keywords['density_fitting'] == True:
            #    if self.keywords['high_level_reference'] == 'rhf':
            #        embedded_wf = DFRMP2(self._mean_field, frozen = frozen_orbitals).run()
            #    elif self.keywords['high_level_reference'] == 'uhf':
            #        embedded_wf = DFUMP2(self._mean_field, frozen = frozen_orbitals).run()
            #else:
            #    embedded_wf = mp.MP2(self._mean_field, frozen = frozen_orbitals).run()
            embedded_wf = mp.MP2(self._mean_field, frozen = frozen_orbitals).run()
            correlation_energy = embedded_wf.e_corr

        if self.keywords['high_level'].lower() == 'fno-mp2':
            correlation_energy = self.custom_FNO_MP2(emb_mf, n_vir_nos, frozen = frozen_orbitals)

        if (self.keywords['high_level'].lower() == 'ccsd' or 
            self.keywords['high_level'].lower() == 'ccsd(t)'):
            embedded_wf = cc.CCSD(self._mean_field).set(frozen = frozen_orbitals).run()
            correlation_energy = embedded_wf.e_corr
            if self.keywords['high_level'].lower() == 'ccsd(t)':
                t_correction = embedded_wf.ccsd_t().T
                correlation_energy += t_correction
        # if span_orbitals provided, store correlation energy of shells
        if span_orbitals is not None:
            self.correlation_energy_shell.append(correlation_energy)
        return correlation_energy

    def effective_virtuals(self):
        """
        Slices the effective virtuals from the entire virtual space.

        Returns
        -------
        effective_orbitals : numpy.array
            Virtual orbitals without the level-shifted orbitals
            from the environment.
        """
        shift = self._n_basis_functions - self.n_env_mos
        effective_orbitals = self._mean_field.mo_coeff[:, self.n_act_mos:shift]
        return effective_orbitals

    def pseudocanonical(self, orbitals):
        """
        Returns pseudocanonical orbitals and the corresponding
        orbital energies.
        
        Parameters
        ----------
        orbitals : numpy.array
            MO coefficients of orbitals to be pseudocanonicalized.

        Returns
        -------
        e_orbital_pseudo : numpy.array
            diagonal elements of the Fock matrix in the
            pseudocanonical basis.
        pseudo_orbitals : numpy.array
            pseudocanonical orbitals.
        """
        fock_matrix = self._mean_field.get_fock()
        mo_fock = orbitals.T @ fock_matrix @ orbitals
        e_orbital_pseudo, pseudo_transformation = np.linalg.eigh(mo_fock)
        pseudo_orbitals = orbitals @ pseudo_transformation
        return e_orbital_pseudo, pseudo_orbitals

    def ao_operator(self):
        """
        Returns the matrix representation of the operator chosen to
        construct the shells.

        Returns
        -------

        K : numpy.array
            Exchange.
        V : numpy.array
            Electron-nuclei potential.
        T : numpy.array
            Kinetic energy.
        H : numpy.array
            Core (one-particle) Hamiltonian.
        S : numpy.array
            Overlap matrix.
        F : numpy.array
            Fock matrix.
        K_orb : numpy.array
            K orbitals (see Feller and Davidson, JCP, 74, 3977 (1981)).
        """
        if (self.keywords['operator'] == 'K' or
            self.keywords['operator'] == 'K_orb'):
            self.operator = self._mean_field.get_k()
            if self.keywords['operator'] == 'K_orb':
                self.operator = 0.06*self._mean_field.get_fock() - self.operator
        elif self.keywords['operator'] == 'V':
            self.operator = self._mol.intor_symmetric('int1e_nuc')
        elif self.keywords['operator'] == 'T':
            self.operator = self._mol.intor_symmetric('int1e_kin')
        elif self.keywords['operator'] == 'H':
            self.operator = self._mean_field.get_hcore()
        elif self.keywords['operator'] == 'S':
            self.operator = self._mean_field.get_ovlp()
        elif self.keywords['operator'] == 'F':
            self.operator = self._mean_field.get_fock()
        return None

class Psi4Embed(Embed):
    """Class with embedding methods using Psi4."""


    def run_mean_field(self, v_emb = None):
        """
        Runs Psi4 (PySCF is coming soon).
        If 'level' is not provided, it runs the a calculation at the level
        given by the 'low_level' key in self.keywords.

        Parameters
        ----------
        v_emb : numpy.array or list of numpy.array (None)
            Embedding potential.
        """
        if v_emb is None:
            self.outfile = open(self.keywords['embedding_output'], 'w')
            # Preparing molecule string with C1 symmetry
            add_c1 = self.keywords['geometry'].splitlines()
            add_c1.append('symmetry c1')
            self.keywords['geometry'] = '\n'.join(add_c1)

            # Running psi4 for the env (low level)
            psi4.set_memory(str(self.keywords['memory']) + ' MB')
            psi4.core.set_num_threads(self.keywords['num_threads'])
            self._mol = psi4.geometry(self.keywords['geometry'])
            self._mol.set_molecular_charge(self.keywords['charge'])
            self._mol.set_multiplicity(self.keywords['multiplicity'])

            psi4.core.be_quiet()
            psi4.core.set_output_file(self.keywords['driver_output'], True)
            psi4.set_options({'save_jk': 'true',
                        'basis': self.keywords['basis'],
                        'reference': self.keywords['low_level_reference'],
                        'ints_tolerance': self.keywords['ints_tolerance'],
                        'e_convergence': self.keywords['e_convergence'],
                        'd_convergence': self.keywords['d_convergence'],
                        'scf_type': self.keywords['eri'],
                        'print': self.keywords['print_level'],
                        'damping_percentage':
                            self.keywords['low_level_damping_percentage'],
                        'soscf': self.keywords['low_level_soscf']
                        })

            self.e, self._wfn = psi4.energy(self.keywords['low_level'],
                molecule = self._mol, return_wfn=True)
            self._n_basis_functions = self._wfn.basisset().nbf()
            if self.keywords['low_level'] != 'HF' :
                self.e_xc_total = psi4.core.VBase.quadrature_values\
                            (self._wfn.V_potential())["FUNCTIONAL"]
                if self.keywords['low_level_reference'] == 'rhf':
                    self.v_xc_total = self._wfn.Va().clone().np
                else:
                    self.alpha_v_xc_total = self._wfn.Va().clone().np
                    self.beta_v_xc_total = self._wfn.Vb().clone().np
            else:
                if self.keywords['low_level_reference'] == 'rhf':
                    #self.v_xc_total = np.zeros([self._n_basis_functions,
                        #self._n_basis_functions])
                    self.v_xc_total = 0.0
                else:
                    #self.alpha_v_xc_total = np.zeros([self._n_basis_functions,
                        #self._n_basis_functions])
                    #self.beta_v_xc_total = np.zeros([self._n_basis_functions,
                        #self._n_basis_functions])
                    self.alpha_v_xc_total = 0.0 
                    self.beta_v_xc_total = 0.0 
                self.e_xc_total = 0.0
        else:
            psi4.set_options({'docc': [self.n_act_mos],
                'reference': self.keywords['high_level_reference']})
            if self.keywords['high_level_reference'] == 'rhf':
                f = open('newH.dat', 'w')
                for i in range(self.h_core.shape[0]):
                    for j in range(self.h_core.shape[1]):
                        f.write("%s\n" % (self.h_core + v_emb)[i, j])
                f.close()
            else:
                psi4.set_options({'socc': [self.n_act_mos - self.beta_n_act_mos]})
                fa = open('Va_emb.dat', 'w')
                fb = open('Vb_emb.dat', 'w')
                for i in range(self.h_core.shape[0]):
                    for j in range(self.h_core.shape[1]):
                        fa.write("%s\n" % v_emb[0][i, j])
                        fb.write("%s\n" % v_emb[1][i, j])
                fa.close()
                fb.close()

            if (self.keywords['high_level'][:2] == 'cc' and
                self.keywords['cc_type'] == 'df'):
                psi4.set_options({'cc_type': self.keywords['cc_type'],
                                'df_ints_io': 'save' })
            self.e, self._wfn = psi4.energy('hf',
                molecule = self._mol, return_wfn=True)

        if self.keywords['low_level_reference'] == 'rhf':
            self.occupied_orbitals = self._wfn.Ca_subset('AO', 'OCC').np
            self.j = self._wfn.jk().J()[0].np
            self.k = self._wfn.jk().K()[0].np
        else:
            self.alpha_occupied_orbitals = self._wfn.Ca_subset('AO', 'OCC').np
            self.beta_occupied_orbitals = self._wfn.Ca_subset('AO', 'OCC').np
            self.alpha_j = self._wfn.jk().J()[0].np
            self.beta_j = self._wfn.jk().J()[1].np
            self.alpha_k = self._wfn.jk().K()[0].np
            self.beta_k = self._wfn.jk().K()[1].np

        self.nre = self._mol.nuclear_repulsion_energy()
        self.ao_overlap = self._wfn.S().np
        self.h_core = self._wfn.H().np
        self.alpha = self._wfn.functional().x_alpha()
        return None

    def count_active_aos(self, basis = None):
        """
        Computes the number of AOs from active atoms.

        Parameters
        ----------
        basis : str
            Name of basis set from which to count active AOs.
        
        Returns
        -------
            self.n_active_aos : int
                Number of AOs in the active atoms.
        """
        if basis is None:
            basis = self._wfn.basisset()
            n_basis_functions = basis.nbf()
        else:
            projected_wfn = psi4.core.Wavefunction.build(self._mol, basis)
            basis = projected_wfn.basisset()
            n_basis_functions = basis.nbf()
            
        self.n_active_aos = 0
        active_atoms = list(range(self.keywords['n_active_atoms']))
        for ao in range(n_basis_functions):
            for atom in active_atoms:
                if basis.function_to_center(ao) == atom:
                   self.n_active_aos += 1
        return self.n_active_aos
        
    def basis_projection(self, orbitals, projection_basis):
        """
        Defines a projection of orbitals in one basis onto another.
        
        Parameters
        ----------
        orbitals : numpy.array
            MO coefficients to be projected.
        projection_basis : str
            Name of basis set onto which orbitals are to be projected.

        Returns
        -------
        projected_orbitals : numpy.array
            MO coefficients of orbitals projected onto projection_basis.
        """
        projected_wfn = psi4.core.Wavefunction.build(self._mol,
            projection_basis)
        mints = psi4.core.MintsHelper(projected_wfn.basisset())
        self.projected_overlap = (
            mints.ao_overlap().np[:self.n_active_aos, :self.n_active_aos])
        self.overlap_two_basis = (mints.ao_overlap(projected_wfn.basisset(),
                            self._wfn.basisset()).np[:self.n_active_aos, :])
        projected_orbitals = (np.linalg.inv(self.projected_overlap)
                            @ self.overlap_two_basis @ orbitals)
        return projected_orbitals

    def closed_shell_subsystem(self, orbitals):
        """
        Computes the potential matrices J, K, and V and subsystem energies.

        Parameters
        ----------
        orbitals : numpy.array
            MO coefficients of subsystem.

        Returns
        -------
        e : float
            Total energy of subsystem.
        e_xc : float
            DFT Exchange-correlation energy of subsystem.
        j : numpy.array
            Coulomb matrix of subsystem.
        k : numpy.array
            Exchange matrix of subsystem.
        v_xc : numpy.array
            Kohn-Sham potential matrix of subsystem.
        """

        density = orbitals @ orbitals.T
        psi4_orbitals = psi4.core.Matrix.from_array(orbitals)

        if hasattr(self._wfn, 'get_basisset'):
            jk = psi4.core.JK.build(self._wfn.basisset(),
                self._wfn.get_basisset('DF_BASIS_SCF'), 'DF')
        else:
            jk = psi4.core.JK.build(self._wfn.basisset())
        jk.set_memory(int(1.25e9))
        jk.initialize()
        jk.C_left_add(psi4_orbitals)
        jk.compute()
        jk.C_clear()
        jk.finalize()

        j = jk.J()[0].np
        k = jk.K()[0].np

        if(self._wfn.functional().name() != 'HF'):
            self._wfn.Da().copy(psi4.core.Matrix.from_array(density))
            self._wfn.form_V()
            v_xc = self._wfn.Va().clone().np
            e_xc = psi4.core.VBase.quadrature_values(
                self._wfn.V_potential())["FUNCTIONAL"]

        else:
            basis = self._wfn.basisset()
            n_basis_functions = basis.nbf()
            v_xc = 0.0
            e_xc = 0.0

        # Energy
        e = self.dot(density, 2.0*(self.h_core + j) - self.alpha*k) + e_xc
        return e, e_xc, 2.0 * j, k, v_xc

    def pseudocanonical(self, orbitals):
        """
        Returns pseudocanonical orbitals and the corresponding
        orbital energies.
        
        Parameters
        ----------
        orbitals : numpy.array
            MO coefficients of orbitals to be pseudocanonicalized.

        Returns
        -------
        e_orbital_pseudo : numpy.array
            diagonal elements of the Fock matrix in the
            pseudocanonical basis.
        pseudo_orbitals : numpy.array
            pseudocanonical orbitals.
        """
        mo_fock = orbitals.T @ self._wfn.Fa().np @ orbitals
        e_orbital_pseudo, pseudo_transformation = np.linalg.eigh(mo_fock)
        pseudo_orbitals = orbitals @ pseudo_transformation
        return e_orbital_pseudo, pseudo_orbitals

    def ao_operator(self):
        """
        Returns the matrix representation of the operator chosen to
        construct the shells.

        Returns
        -------

        K : numpy.array
            Exchange.
        V : numpy.array
            Electron-nuclei potential.
        T : numpy.array
            Kinetic energy.
        H : numpy.array
            Core (one-particle) Hamiltonian.
        S : numpy.array
            Overlap matrix.
        F : numpy.array
            Fock matrix.
        K_orb : numpy.array
            K orbitals (see Feller and Davidson, JCP, 74, 3977 (1981)).
        """
        if (self.keywords['operator'] == 'K' or
            self.keywords['operator'] == 'K_orb'):
            jk = psi4.core.JK.build(self._wfn.basisset(),
                self._wfn.get_basisset('DF_BASIS_SCF'),'DF')
            jk.set_memory(int(1.25e9))
            jk.initialize()
            jk.print_header()
            jk.C_left_add(self._wfn.Ca())
            jk.compute()
            jk.C_clear()
            jk.finalize()
            self.operator = jk.K()[0].np
            if self.keywords['operator'] == 'K_orb':
                self.operator = 0.06*self._wfn.Fa().np - self.K
        elif self.keywords['operator'] == 'V':
            mints = psi4.core.MintsHelper(self._wfn.basisset())
            self.operator = mints.ao_potential().np
        elif self.keywords['operator'] == 'T':
            mints = psi4.core.MintsHelper(self._wfn.basisset())
            self.operator = mints.ao_kinetic().np
        elif self.keywords['operator'] == 'H':
            self.operator = self._wfn.H().np
        elif self.keywords['operator'] == 'S':
            self.operator = self._wfn.S().np
        elif self.keywords['operator'] == 'F':
            self.operator = self._wfn.Fa().np

    def open_shell_subsystem(self, alpha_orbitals, beta_orbitals):
        """
        Computes the potential matrices J, K, and V and subsystem
        energies for open shell cases.

        Parameters
        ----------
        alpha_orbitals : numpy.array
            Alpha MO coefficients.
        beta_orbitals : numpy.array
            Beta MO coefficients.

        Returns
        -------
        e : float
            Total energy of subsystem.
        e_xc : float
            Exchange-correlation energy of subsystem.
        alpha_j : numpy.array
            Alpha Coulomb matrix of subsystem.
        beta_j : numpy.array
            Beta Coulomb matrix of subsystem.
        alpha_k : numpy.array
            Alpha Exchange matrix of subsystem.
        beta_k : numpy.array
            Beta Exchange matrix of subsystem.
        alpha_v_xc : numpy.array
            Alpha Kohn-Sham potential matrix of subsystem.
        beta_v_xc : numpy.array
            Beta Kohn-Sham potential matrix of subsystem.
        """
        alpha_density = alpha_orbitals @ alpha_orbitals.T
        beta_density = beta_orbitals @ beta_orbitals.T

        # J and K
        jk = psi4.core.JK.build(self._wfn.basisset(),
            self._wfn.get_basisset('DF_BASIS_SCF'), 'DF')
        jk.set_memory(int(1.25e9))
        jk.initialize()
        jk.C_left_add(psi4.core.Matrix.from_array(alpha_orbitals))
        jk.C_left_add(psi4.core.Matrix.from_array(beta_orbitals))
        jk.compute()
        jk.C_clear()
        jk.finalize()
        alpha_j = jk.J()[0].np
        beta_j = jk.J()[1].np
        alpha_k = jk.K()[0].np
        beta_k = jk.K()[1].np
        
        if(self._wfn.functional().name() != 'HF'):
            self._wfn.Da().copy(psi4.core.Matrix.from_array(alpha_density))
            self._wfn.Db().copy(psi4.core.Matrix.from_array(beta_density))
            self._wfn.form_V()
            alpha_v_xc = self._wfn.Va().clone().np
            beta_v_xc = self._wfn.Vb().clone().np
            e_xc = psi4.core.VBase.quadrature_values(
                self._wfn.V_potential())['FUNCTIONAL']
        else:
            #alpha_v_xc = np.zeros([self._n_basis_functions,
                #self._n_basis_functions])
            #beta_v_xc = np.zeros([self._n_basis_functions,
                #self._n_basis_functions])
            alpha_v_xc = 0.0
            beta_v_xc = 0.0
            e_xc = 0.0

        e = (self.dot(self.h_core, alpha_density + beta_density)
            + 0.5*(self.dot(alpha_j + beta_j, alpha_density + beta_density)
            - self.alpha*self.dot(alpha_k, alpha_density)
            - self.alpha*self.dot(beta_k, beta_density)) + e_xc)

        return e, e_xc, alpha_j, beta_j, alpha_k, beta_k, alpha_v_xc, beta_v_xc

    def orthonormalize(self, S, C, n_non_zero):
        """
        (Deprecated) Orthonormalizes a set of orbitals (vectors).

        Parameters
        ----------
        S : numpy.array
            Overlap matrix in AO basis.
        C : numpy.array
            MO coefficient matrix, vectors to be orthonormalized.
        n_non_zero : int
            Number of orbitals that have non-zero norm.

        Returns
        -------
        C_orthonormal : numpy.array
            Set of n_non_zero orthonormal orbitals.
        """

        overlap = C.T @ S @ C
        v, w = np.linalg.eigh(overlap)
        idx = v.argsort()[::-1]
        v = v[idx]
        w = w[:,idx]
        C_orthonormal = C @ w
        for i in range(n_non_zero):
            C_orthonormal[:,i] = C_orthonormal[:,i]/np.sqrt(v[i])
        return C_orthonormal[:,:n_non_zero]

    def molden(self, shell_orbitals, shell):
        """
        Creates molden file from orbitals at the shell.

        Parameters
        ----------
        span_orbitals : numpy.array
            Span orbitals.
        shell : int
            Shell index.
        """
        self._wfn.Ca().copy(psi4.core.Matrix.from_array(shell_orbitals))
        psi4.driver.molden(self._wfn, str(shell) + '.molden')
        return None

    def heatmap(self, span_orbitals, kernel_orbitals, shell):
        """
        Creates heatmap file from orbitals at the i-th shell.

        Parameters
        ----------
        span_orbitals : numpy.array
            Span orbitals.
        kernel_orbitals : numpy.array
            Kernel orbitals.
        shell : int
            Shell index.
        """
        orbitals = np.hstack((span_orbitals, kernel_orbitals))
        mo_operator = orbitals.T @ self.operator @ orbitals
        np.savetxt('heatmap_'+str(shell)+'.dat', mo_operator)
        return None

    def correlation_energy(self, span_orbitals = None, kernel_orbitals = None,
        span_orbital_energies = None, kernel_orbital_energies = None):
        """
        Computes the correlation energy for the current set of active
        virtual orbitals.
        
        Parameters
        ----------
        span_orbitals : numpy.array
            Orbitals transformed by the span of the previous shell.
        kernel_orbitals : numpy.array
            Orbitals transformed by the kernel of the previous shell.
        span_orbital_energies : numpy.array
            Orbitals energies of the span orbitals.
        kernel_orbital_energies : numpy.array
            Orbitals energies of the kernel orbitals.

        Returns
        -------
        correlation_energy : float
            Correlation energy of the span_orbitals.
        """
        shift = self._n_basis_functions - self.n_env_mos
        if span_orbitals is None:
            nfrz = self.n_env_mos
        else:
            effective_orbitals = np.hstack((span_orbitals,
                kernel_orbitals))
            orbital_energies = np.concatenate((span_orbital_energies,
                kernel_orbital_energies))
            nfrz = (self._n_basis_functions - self.n_act_mos
                 - span_orbitals.shape[1])
            orbitals = np.hstack((self.occupied_orbitals,
                effective_orbitals, self._wfn.Ca().np[:, shift:]))
            orbital_energies = (
                np.concatenate((self._wfn.epsilon_a().np[:self.n_act_mos],
                orbital_energies, self._wfn.epsilon_a().np[shift:])))
            self._wfn.Ca().copy(psi4.core.Matrix.from_array(orbitals))
            self._wfn.epsilon_a().np[:] = orbital_energies[:]

        # Update the number of frozen orbitals and compute energy
        frzvpi = psi4.core.Dimension.from_list([nfrz])
        self._wfn.new_frzvpi(frzvpi)
        #wf_eng, wf_wfn = psi4.energy(self.keywords['high_level'],
            #ref_wfn = self._wfn, return_wfn = True)
        psi4.energy(self.keywords['high_level'], ref_wfn = self._wfn)
        correlation_energy = psi4.core.get_variable(
            self.keywords['high_level'].upper() + " CORRELATION ENERGY")
        self.correlation_energy_shell.append(correlation_energy)
        return correlation_energy

    def effective_virtuals(self):
        """
        Slices the effective virtuals from the entire virtual space.

        Returns
        -------
        effective_orbitals : numpy.array
            Virtual orbitals without the level-shifted orbitals
            from the environment.
        """
        shift = self._n_basis_functions - self.n_env_mos
        effective_orbitals = self._wfn.Ca().np[:, self.n_act_mos:shift]
        return effective_orbitals

