"""Compare multiple discrete MBH Binary Populations (from cosmological hydrodynamic simulations)."""

import os
import sys
import h5py
import numpy as np
import pickle
import holodeck as holo
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import warnings

import holodeck.sams
import holodeck.gravwaves
from holodeck import cosmo, utils, log, plot, discrete, sams, host_relations, _PATH_DATA
from holodeck.constants import MSOL, PC, YR, MPC, GYR, SPLC


class Discrete:
    
    def __init__(self, attrs=(None,None,'k',1.0), lbl=None, fixed_sepa=None, 
                 tau=None, rchar=None, gamma_inner=None, gamma_outer=None, hard_model_type='old_rc100',
                 nreals=500, mod_mmbulge=False, rescale_mbulge=False, allow_mbh0=False, 
                 skip_evo=False, subhalo_mstar_defn='SubhaloMassInRadType', bfrac=None, nloudest=10,
                 plot_mtot_vs_q=False, apply_mtot_mrat_cuts=False):

        self.attrs = attrs
        self.lbl = lbl
        self.fname = self.attrs[0]
        self.basepath = self.attrs[1]
        self.minNparts = self.attrs[2]
        self.color = self.attrs[3]
        self.lw = self.attrs[4]
        self.fixed_sepa = fixed_sepa
        self.nreals = nreals
        self.mod_mmbulge = mod_mmbulge
        self.allow_mbh0 = allow_mbh0
        #self.use_mstar_tot_as_mbulge = use_mstar_tot_as_mbulge
        self.subhalo_mstar_defn = subhalo_mstar_defn
        self.bfrac = bfrac
        self.nloudest = nloudest
        
        print(f"\nCreating Discrete_Pop class instance '{self.lbl}' with {self.fixed_sepa=}, {tau=} & {hard_model_type=}.")
        print(f" fname={self.fname}")
        self.pop = discrete.population.Pop_Illustris(fname=self.fname, basepath=self.basepath, 
                                                     fixed_sepa=self.fixed_sepa, allow_mbh0=self.allow_mbh0,
                                                     subhalo_mstar_defn=self.subhalo_mstar_defn, bfrac=self.bfrac)
        print(f"{self.pop.sepa.min()=}, {self.pop.sepa.max()=}, {self.pop.sepa.shape=}")
        #print(f"{self.pop.mstar_tot.min()=}, {self.pop.mstar_tot.max()=}, {self.pop.mstar_tot.shape=}")
        print(f"{self.pop.mstar.min()=}, {self.pop.mstar.max()=}, {self.pop.mstar.shape=}")

        self.sim_mass_resolution()
        
        # apply modifiers if requested
        if self.mod_mmbulge == True:
            print(f"before mass mod: {self.pop.mass.min()=}, {self.pop.mass.max()=}, {self.pop.mass.shape=}")
            print(f"before mass mod: {self.pop.mbulge.min()=}, {self.pop.mbulge.max()=}, {self.pop.mbulge.shape=}")
            old_mass = self.pop.mass
            old_mbulge = self.pop.mbulge
            old_mrat = self.pop.mass[:,1]/self.pop.mass[:,0]
            old_mrat[old_mrat>1] = 1/old_mrat[old_mrat>1]
            
            print(f"before mass mod: mass ratio m2/m1: {old_mrat.min()=}, {old_mrat.max()=}, {old_mrat.shape=}")
            ## self.mmbulge = holo.relations.MMBulge_KH2013() # deprecated
            self.mmbulge = holo.host_relations.MMBulge_KH2013()
            self.mod_KH2013 = discrete.population.PM_Mass_Reset(self.mmbulge, scatter=True, 
                                                                rescale_mbulge=rescale_mbulge)
            self.pop.modify(self.mod_KH2013)

            # ---- Added for debugging change in mass ratios 5/15/25 - LB ---- #
            if plot_mtot_vs_q:
                new_mrat = self.pop.mass[:,1]/self.pop.mass[:,0]
                new_mrat[new_mrat>1] = 1/new_mrat[new_mrat>1]

                print(f"after mass mod: {self.pop.mass.min()=}, {self.pop.mass.max()=}, {self.pop.mass.shape=}")
                print(f"after mass mod: {self.pop.mbulge.min()=}, {self.pop.mbulge.max()=}, {self.pop.mbulge.shape=}")
                print(f"after mass mod: mass ratio m2/m1: {new_mrat.min()=}, {new_mrat.max()=}, {new_mrat.shape=}")

                mrat_increase_factor = new_mrat / old_mrat
                mass_increase_factor = self.pop.mass / old_mass
                mbulge_increase_factor = self.pop.mbulge / old_mbulge
                test_old_mass_fac = self.pop._mass / old_mass
                print(f"after mass mod: {mrat_increase_factor.min()=}, {mrat_increase_factor.max()=},{np.median(mrat_increase_factor)=}")
                print(f"after mass mod: {mass_increase_factor.min()=}, {mass_increase_factor.max()=}, {np.median(mass_increase_factor)=}")
                print(f"after mass mod: {mbulge_increase_factor.min()=}, {mbulge_increase_factor.max()=}, {np.median(mbulge_increase_factor)=}")
                print(f"after mass mod: {test_old_mass_fac.min()=}, {test_old_mass_fac.max()=}, {np.median(test_old_mass_fac)=}")

                ix_low_mrat = np.where(mrat_increase_factor<0.25)[0]
                print(f"{ix_low_mrat.size=}")

                plt.xscale('log')
                plt.yscale('log')
                plt.xlabel('q')
                plt.ylabel('Mtot [Msun]')
                old_mtot = old_mass[:,0] + old_mass[:,1]
                new_mtot = self.pop.mass[:,0] + self.pop.mass[:,1]
                himass_mrat_increased = 0
                himass_mrat_decreased = 0
                mrat_increased = 0
                mrat_decreased = 0
                himass_count = 0
                for i in range(old_mrat.size):
                    if new_mrat[i]>old_mrat[i]:
                        mrat_increased += 1
                    else:
                        mrat_decreased += 1
                    
                    if np.max([old_mass[i,0],old_mass[i,1],self.pop.mass[i,0],self.pop.mass[i,1]])>1e8*MSOL:
                        himass_count += 1
                        if new_mrat[i]>old_mrat[i]:
                            col='r'
                            himass_mrat_increased += 1 
                        else: 
                            col='k'
                            himass_mrat_decreased += 1
                        plt.plot([old_mrat[i],new_mrat[i]], [old_mtot[i]/MSOL,new_mtot[i]/MSOL],alpha=0.3, lw=0.5, color=col)
                        plt.plot([new_mrat[i]], [new_mtot[i]/MSOL],alpha=0.3, marker='.', ms=2, color=col)
                print(f"{old_mrat.size=}, {mrat_increased=}, {mrat_decreased=}")
                print(f"{himass_count=}, {himass_mrat_increased=}, {himass_mrat_decreased=}")
                plt.show()
            # ---------------------------------------------------------------------------- #

        if apply_mtot_mrat_cuts:
            print(f"In Discrete class")
            print(f"Current mass range: {self.pop.mass.min()}-{self.pop.mass.max()}")
            #print(f"Current mrat range: {self.pop.mrat.min()}-{self.pop.mrat.max()}")
            print(f"Applying mtot and mrat cuts to current population of {self.pop.mass.shape} binaries.")
            self.mod_mcuts = discrete.population.PM_Cuts(mrat_min=0.001, mtot_min=1.0e4*MSOL)
            self.pop.modify(self.mod_mcuts)
            print(f"After applying mrat and mtot cuts, population has {self.pop.mass.shape} binaries.")

            
        if skip_evo == False:
            # create a fixed-total-time hardening mechanism
            
            if hard_model_type is not None:
                self.hard_model_type = hard_model_type
                print(f'creating fixed-time hardening using pre-defined model {self.hard_model_type}.')
                self.set_sim_hard_params_manual(tau=tau)
                
            else:
                if None in (tau, rchar, gamma_inner, gamma_outer):
                    raise ValueError('if keyword `hard_model_type` is None, numerical values are required for:\n'
                                     '    `tau` [Gyr], `rchar` [pc], `gamma_inner`, & `gamma_outer`.')
                else:
                    self.hard_model_type = 'kw_input'
                    print(f'creating fixed-time hardening using kw input: '
                          f'{tau=}, {rchar=}, {gamma_inner=}, {gamma_outer=}.')
                    self.HARD_PARS = dict(
                        desc='from keyword input',
                        hard_time = tau,    # [pc]
                        hard_rchar = rchar,    # [pc]
                        hard_gamma_inner = gamma_inner,
                        hard_gamma_outer = gamma_outer
                    )

            print(f"modeling fixed-total-time hardening...")
            #self.fixed = holo.hardening.Fixed_Time_2PL.from_pop(self.pop, self.tau)
            self.fixed = holo.hardening.Fixed_Time_2PL.from_pop(self.pop, self.HARD_PARS['hard_time']*GYR, 
                                                                rchar=self.HARD_PARS['hard_rchar']*PC, 
                                                                gamma_inner=self.HARD_PARS['hard_gamma_inner'], 
                                                                gamma_outer=self.HARD_PARS['hard_gamma_outer'])

            print(f"{self.pop.sepa.min()=}, {self.pop.sepa.max()=}, {self.pop.sepa.shape=}")

            # Create an evolution instance using population and hardening mechanism
            print(f"creating evolution instance and evolving it...")
            self.evo = discrete.evolution.Evolution(self.pop, self.fixed)
            #self.evo = discrete.evolution.Evolution(self.pop, self.fixed, debug=True)
            print(f"{self.evo._sample_volume=}")
            print(f"{self.pop.scafa.min()=}, {self.pop.scafa.max()=}, {self.pop.scafa.shape=}")

            # evolve binary population
            self.evo.evolve()
            coal = self.evo.coal
            print(f"{coal.shape=}, {coal[coal].shape}")
            print(f"{self.evo.mass.shape=}")
            
            ## create GWB
            self.freqs, self.freqs_edges = utils.pta_freqs()
            self.gwb = holo.gravwaves.GW_Discrete(self.evo, self.freqs, nreals=self.nreals) #, nloudest=self.nloudest)
            self.gwb.emit(nloudest=self.nloudest)

            
            
    def set_sim_hard_params_manual(self, tau=None):

        if tau is None:
            raise ValueError("must choose a numerical value of keyword `tau` (in Gyr)!")
        
        # ---- Define the hardening model params
        # note that intial binary separation is set separately in population.py
        if self.hard_model_type == 'old_rc10':
            self.HARD_PARS = dict(
                desc='LB old w rchar=10',
                hard_rchar=10.0,        # [pc]
                hard_gamma_inner=-1.0,
                hard_gamma_outer=+1.5,
            )
        elif self.hard_model_type == 'old_rc100':
            self.HARD_PARS = dict(
                desc='LB old w rchar=100',
                hard_rchar=100.0,        # [pc]
                hard_gamma_inner=-1.0,
                hard_gamma_outer=+1.5,
            )
        elif self.hard_model_type == 'old_rc10_nu0':
            self.HARD_PARS = dict(
                desc='LB old w rchar=10 & nu_out=0',
                hard_rchar=10.0,        # [pc]
                hard_gamma_inner=-1.0,
                hard_gamma_outer=0.0,
            )
        elif self.hard_model_type == 'ph15':
            self.HARD_PARS = dict(
                desc='Phenom 15yr',
                hard_rchar=100.0,        # [pc]
                hard_gamma_inner=-1.0,
                hard_gamma_outer=+2.5,
            )
        elif self.hard_model_type == 'ph15_rc10':
            self.HARD_PARS = dict(
                desc='Phenom 15yr',
                hard_rchar=10.0,        # [pc]
                hard_gamma_inner=-1.0,
                hard_gamma_outer=+2.5,
            )
        else:
            modlist = ['old_rc10', 'old_rc100', 'old_rc10_nu0', 'ph15', 'ph15_rc10']

            raise ValueError(f"{self.hard_model_type=} is not defined. Options are {[m for m in modlist]}.")

        # define the fixed inspiral timescale from input keyword
        self.HARD_PARS["hard_time"] = tau    # [Gyr]

        print(f"Set SAM params manually for {self.hard_model_type=}.")
    

    def sim_mass_resolution(self):
        # Baryonic mass resolution for each simulation, in Msun
        mres_baryon = {
            'Illustris-1': 1.3e6,
            'TNG50-1': 8.4e4,
            'TNG50-2': 6.8e5,
            'TNG50-3': 5.4e6,
            'TNG100-1': 1.4e6,
            'TNG100-2': 1.1e7,
            'TNG300-1': 1.1e7,
        }
        self.mres_baryon = None
        for k in mres_baryon.keys():
            if k in self.lbl: 
                self.mres_baryon = mres_baryon[k]
                break
        if self.mres_baryon is None:
            raise ValueError(f"{self.lbl=} has no match for mres_baryon.keys().")

    
    def load_sim_gsmf_file(self, basePath=_PATH_DATA):

        for s in ['Illustris-1', 'TNG50-1', 'TNG100-1', 'TNG300-1']:
            if s in self.lbl: 
                print(f"{s=}")
                try:
                    gsmf_fpath = os.path.join(basePath, f"gsmf_all_snaps_{s}.hdf5")
                    f = h5py.File(gsmf_fpath,"r")
                    print(f"opened {gsmf_fpath=}")
                except: 
                    try:
                        gsmf_fpath = os.path.join(_PATH_DATA, f"gsmf_all_snaps_{s}.hdf5")
                        f = h5py.File(gsmf_fpath,"r")
                        print(f"opened {gsmf_fpath=}")
                    except:
                        raise Exception(f"Could not open GSMF file {gsmf_fpath}.")

                break

        if not hasattr(self, "mhist_bins"): self.mhist_bins = {}
        if not hasattr(self, "gsmf"): self.gsmf = {}
        if not hasattr(self, "mhist"): self.mhist = {}
        
        return f
        
    def calc_sim_gsmf_from_snaps(self, req_z, req_binsize=0.05, verbose=False): 

        f = self.load_sim_gsmf_file(self.basepath)
        try:
            grpname=(f"gas-{self.minNparts[0]:03d}_dm-{self.minNparts[1]:03d}_"
                     f"star-{self.minNparts[4]:03d}_bh-{self.minNparts[5]:03d}")
            grp=f[grpname]
        except:
            raise Exception(f"Could not open group in GSMF file: {grpname}")


        box_vol_mpc = f.attrs['box_volume_mpc']
        snapnums = f.attrs['SnapshotNums']
        scalefacs = f.attrs['SnapshotScaleFacs']
        zsnaps = 1.0 / scalefacs - 1.0
    
        diff = np.abs(zsnaps-req_z)
        snapNum = snapnums[diff==diff.min()][0]
        zsnap = zsnaps[diff==diff.min()][0]
        if verbose or (diff.min()>0.01):
            print(f"{req_z=}, {snapNum=}, {zsnap=}, {diff.min()=}")

        dlgm_orig = f.attrs['LogMassBinWidth']
        mbin_edges_orig = np.array(grp['StellarMassBinEdges'])
        nbins_orig = mbin_edges_orig.size - 1
        mhist_all_snaps = np.array(grp['StellarMassHistograms'])
    
        mhist_snap_orig = mhist_all_snaps[:,(snapnums==snapNum)].flatten()
        if verbose: print(f"{mhist_snap_orig.shape=}, {mbin_edges_orig.shape=}")
        if mhist_snap_orig.size != nbins_orig:
            print('whoops')
            return

        if req_binsize < dlgm_orig:
            raise ValueError(f"{req_binsize=} requested, but min allowed is {dlgm_orig=}")
        if int(req_binsize/dlgm_orig) > nbins_orig/2:
            raise ValueError(f"{req_binsize=} requested, but max allowed is {dlgm_orig*nbins_orig/2=}")

        ncomb = int(req_binsize/dlgm_orig)
        dlgm = dlgm_orig * ncomb
        mbin_edges = mbin_edges_orig[::ncomb]
        nbins = mbin_edges.size
        if ncomb > 1:
            mbin_edges = np.append(mbin_edges, mbin_edges[-1]+dlgm)
            mhist_snap = np.zeros((nbins))
            if verbose: print(f"{mbin_edges.size=}")
            for i in range(mbin_edges.size-1):
                mhist_snap[i] = mhist_snap_orig[i*ncomb:i*ncomb+ncomb].sum()
            if verbose:
                print(f"{mbin_edges_orig=}")
                print(f"{mbin_edges=}")
        else:
            if verbose:
                print(f"WARNING: {req_binsize=}, {ncomb=}; retaining original binsize {dlgm_orig=}")
            assert mbin_edges.all() == mbin_edges_orig.all() and dlgm == dlgm_orig, "Error in setting ncomb=1!"
            mhist_snap = mhist_snap_orig
        
        if verbose:
            print(f"{mhist_all_snaps.shape=}, {mhist_all_snaps.min()=}, {mhist_all_snaps.max()=}")
            print(f"{mhist_snap.shape=}, {mhist_snap.min()=}, {mhist_snap.max()=}")
            print(f"{snapnums=}")
            print(f"{dlgm=}, {mbin_edges.shape=}")
            print(f"{mbin_edges=}")

        ##gsmf = mhist_snap / dlgm / np.log(10) / box_vol_mpc  # dex^-1 Mpc^-3
        gsmf = mhist_snap / dlgm / box_vol_mpc  # dex^-1 Mpc^-3

        self.mhist_bins[req_z] = mbin_edges[:-1]+0.5*dlgm
        self.gsmf[req_z] = gsmf
        self.mhist[req_z] = mhist_snap
        
        #return mbin_edges[:-1]+0.5*dlgm, gsmf, mhist_snap #mbin_edges, dlgm



def create_dpops(tau=None, rchar=None, gamma_inner=None, gamma_outer=None, hard_model_type=None,
                 fsa=1.0e4, mod_mmbulge=True, nreals=500, nloudest=10, 
                 inclIll=False, inclOldIll=False, inclT50=False, inclT100=False, inclT300=False, 
                 inclRescale=False, allow_mbh0=False, skip_evo=False, fsa_only=False, 
                 apply_mtot_mrat_cuts=False,subhalo_mstar_defn='MaxPastMass',
                 bfrac=None, fpath=_PATH_DATA, pickle_dpops=True, pickle_name=''):
    
    if (fsa is None) and (fsa_only):
        raise ValueError(f"No dpops to create! {fsa_only=} and {fsa=}.")
    
    print(np.array([inclIll, inclOldIll, inclT50, inclT100, inclT300]))
    if not np.any(np.array([inclIll, inclOldIll, inclT50, inclT100, inclT300])): 
        raise ValueError("No dpops to create! At least one of `inclIll`, `inclOldIll`, "
                         "`inclT50`, `inclT100`, `inclT300` must be True.")
    
    if hard_model_type is None: 
        if None in (tau, rchar, gamma_inner, gamma_outer):
            raise ValueError('if keyword `hard_model_type` is None, numerical values are required for:\n'
                             '    `tau` [Gyr], `rchar` [pc], `gamma_inner`, & `gamma_outer`.')
        if pickle_dpops:
            pickle_name += f'_t{tau}rc{rchar}nin{gamma_inner}nout{gamma_outer}'

    else:
        if rchar is not None:
            print(f"WARNING: `hard_model_type` is not None, but keyword {rchar=}. resetting `rchar`=None.")
            rchar = None
        if gamma_inner is not None:
            print(f"WARNING: `hard_model_type` is not None, but keyword {gamma_inner=}. resetting `gamma_inner`=None.")
            gamma_inner = None
        if gamma_outer is not None:
            print(f"WARNING: `hard_model_type` is not None, but keyword {gamma_outer=}. resetting `gamma_outer`=None.")
            gamma_outer = None
        
        if pickle_dpops:
            pickle_name += f'_hard{hard_model_type}_tau{tau}'

    
    ## ---- Define the GWB frequencies if not already defined
    #if freqs is None or freqs_edges is None:
    #    if freqs is not None or freqs_edges is not None:
    #        msg = ('`freqs` and `freqs_edges` must both be defined to use input frequencies.'
    #               ' Ignoring input values and loading frequencies from utils.pta_freqs() instead.')
    #        log.warning(msg)
    #    freqs, freqs_edges = utils.pta_freqs()
    #    
    #nfreqs = freqs.shape[0]
    
    # ---- Initialize return variables
    all_dpops = []
    tng_dpops = []

    # ---- (Optionally) set the fixed initial binary separation & initialize fsa return vars
    if fsa is not None:
        print(f"Setting fixed init binary sep = {fsa} pc.")
        fsa = fsa * PC
        all_fsa_dpops = []
        tng_fsa_dpops = []
        
    # ---- Define dpop attributes: (filename, filepath, minNparts, plot color, plot linewidth)    
    #tpath = '/orange/lblecha/IllustrisTNG/Runs/'
    #ipath = '/orange/lblecha/Illustris/'
    

    ill_dpop_attrs = {
        'Illustris-1-N001-bh1' : ('galaxy-mergers_Illustris-1_gas-000_dm-000_star-001_bh-001.hdf5', fpath,
                                  np.array([0,0,0,0,1,1]).astype('int64'), 'darkgreen', 1.0),
        'Illustris-1-N010-bh0' : ('galaxy-mergers_Illustris-1_gas-000_dm-000_star-010_bh-000.hdf5', fpath,
                                  np.array([0,0,0,0,10,0]).astype('int64'), 'darkgreen', 1.5),
        'Illustris-1-N010-bh1' : ('galaxy-mergers_Illustris-1_gas-000_dm-000_star-010_bh-001.hdf5', fpath,
                                  np.array([0,0,0,0,10,1]).astype('int64'), 'darkgreen', 2.0),
        'Illustris-1-bh0' : ('galaxy-mergers_Illustris-1_gas-100_dm-100_star-100_bh-000.hdf5', fpath, 
                             np.array([100,100,0,0,100,0]).astype('int64'), 'g', 2.25),
        'Illustris-1' : ('galaxy-mergers_Illustris-1_gas-100_dm-100_star-100_bh-001.hdf5', fpath, 
                         np.array([100,100,0,0,100,1]).astype('int64'), 'g', 2.5)
    }
        
    tng50_dpop_attrs = {       
        'TNG50-1-N100' : ('galaxy-mergers_TNG50-1_gas-100_dm-100_star-100_bh-001.hdf5',  fpath, 
                          np.array([100,100,0,0,100,1]).astype('int64'), 'darkred', 4),
        'TNG50-1-N100-bh0' : ('galaxy-mergers_TNG50-1_gas-100_dm-100_star-100_bh-000.hdf5', fpath, 
                              np.array([100,100,0,0,100,0]).astype('int64'), 'darkred', 3),
        'TNG50-1-bh0' : ('galaxy-mergers_TNG50-1_gas-800_dm-800_star-800_bh-000.hdf5', fpath, 
                         np.array([800,800,0,0,800,0]).astype('int64'), 'r', 2.5),
        'TNG50-1' : ('galaxy-mergers_TNG50-1_gas-800_dm-800_star-800_bh-001.hdf5', fpath, 
                     np.array([800,800,0,0,800,1]).astype('int64'), 'r', 3.5),
        'TNG50-2' : ('galaxy-mergers_TNG50-2_gas-100_dm-100_star-100_bh-001.hdf5', fpath, 
                     np.array([100,100,0,0,100,1]).astype('int64'), 'orange', 2.5),
        'TNG50-3' : ('galaxy-mergers_TNG50-3_gas-012_dm-012_star-012_bh-001.hdf5', fpath, 
                     np.array([12,12,0,0,12,1]).astype('int64'), 'y', 1.5)
    }

    tng100_dpop_attrs = {
        'TNG100-1-N010-bh0' : ('galaxy-mergers_TNG100-1_gas-000_dm-000_star-010_bh-000.hdf5', fpath, 
                               np.array([0,0,0,0,10,0]).astype('int64'), 'darkblue', 2.5),
        'TNG100-1-bh0' : ('galaxy-mergers_TNG100-1_gas-100_dm-100_star-100_bh-000.hdf5', fpath, 
                          np.array([100,100,0,0,100,0]).astype('int64'), 'b', 1.5),
        'TNG100-1' : ('galaxy-mergers_TNG100-1_gas-100_dm-100_star-100_bh-001.hdf5', fpath, 
                      np.array([100,100,0,0,100,1]).astype('int64'), 'b', 2.5),
        'TNG100-2' : ('galaxy-mergers_TNG100-2_gas-012_dm-012_star-012_bh-001.hdf5', fpath, 
                      np.array([12,12,0,0,12,1]).astype('int64'), 'c', 1.5)
    }

    # the N100 cuts here aren't consistent with other sims, they just produce smaller more manageable files
    tng300_dpop_attrs = {
        # temporarily commenting out b/c not finished running code to generate file
        #'TNG300-1-bh0' : ('galaxy-mergers_TNG300-1_gas-012_dm-012_star-012_bh-000.hdf5', fpath, 
        #                  np.array([12,12,0,0,12,0]).astype('int64'), 'm', 1.0),
        'TNG300-1' : ('galaxy-mergers_TNG300-1_gas-012_dm-012_star-012_bh-001.hdf5', fpath, 
                      np.array([12,12,0,0,12,1]).astype('int64'), 'm', 1.5),
        #'TNG300-1-N100' : ('galaxy-mergers_TNG300-1_gas-100_dm-100_star-100_bh-001.hdf5', fpath, 
        #                   np.array([100,100,0,0,100,1]).astype('int64'), 'pink', 1.5),
        #'TNG300-1-N100-bh0' : ('galaxy-mergers_TNG300-1_gas-100_dm-100_star-100_bh-000.hdf5', fpath, 
        #                       np.array([100,100,0,0,100,0]).astype('int64'), 'pink', 1)
    }

    # ---- Create dict of dpops for specified sims
    dpop_attrs = {}
    if inclIll:
        dpop_attrs.update(ill_dpop_attrs)
    if inclT50:
        dpop_attrs.update(tng50_dpop_attrs)
    if inclT100:
        dpop_attrs.update(tng100_dpop_attrs)
    if inclT300:
        dpop_attrs.update(tng300_dpop_attrs)

    
    #dpop_attrs = {
    #    'Illustris-1-N001-bh1' : ('galaxy-mergers_Illustris-1_gas-000_dm-000_star-001_bh-001.hdf5', fpath,
    #                              np.array([0,0,0,0,1,1]).astype('int64'), 'darkgreen', 1.0),
    #    'Illustris-1-N010-bh0' : ('galaxy-mergers_Illustris-1_gas-000_dm-000_star-010_bh-000.hdf5', fpath,
    #                              np.array([0,0,0,0,10,0]).astype('int64'), 'darkgreen', 1.5),
    #    'Illustris-1-N010-bh1' : ('galaxy-mergers_Illustris-1_gas-000_dm-000_star-010_bh-001.hdf5', fpath,
    #                              np.array([0,0,0,0,10,1]).astype('int64'), 'darkgreen', 2.0),
    #    'Illustris-1-bh0' : ('galaxy-mergers_Illustris-1_gas-100_dm-100_star-100_bh-000.hdf5', fpath, 
    #                         np.array([100,100,0,0,100,0]).astype('int64'), 'g', 2.25),
    #    'Illustris-1' : ('galaxy-mergers_Illustris-1_gas-100_dm-100_star-100_bh-001.hdf5', fpath, 
    #                     np.array([100,100,0,0,100,1]).astype('int64'), 'g', 2.5),
    #    #'TNG50-1-N100' : ('galaxy-mergers_TNG50-1_gas-100_dm-100_star-100_bh-001.hdf5',  fpath, 'darkred', 4),
    #    #'TNG50-1-N100-bh0' : ('galaxy-mergers_TNG50-1_gas-100_dm-100_star-100_bh-000.hdf5', fpath, 'darkred', 3),
    #    #'TNG50-1-bh0' : ('galaxy-mergers_TNG50-1_gas-800_dm-800_star-800_bh-000.hdf5', fpath, 'r', 2.5),
    #    #'TNG50-1' : ('galaxy-mergers_TNG50-1_gas-800_dm-800_star-800_bh-001.hdf5', fpath, 
    #    #             np.array([800,800,0,0,800,1]).astype('int64')'r', 3.5),
    #    #'TNG50-2' : ('galaxy-mergers_TNG50-2_gas-100_dm-100_star-100_bh-001.hdf5', fpath, 'orange', 2.5),
    #    #'TNG50-3' : ('galaxy-mergers_TNG50-3_gas-012_dm-012_star-012_bh-001.hdf5', fpath, 'y', 1.5),
    #    #'TNG100-1-N010-bh0' : ('galaxy-mergers_TNG100-1_gas-000_dm-000_star-010_bh-000.hdf5', fpath, 
    #    #                       np.array([0,0,0,0,10,0]).astype('int64'), 'darkblue', 2.5),
    #    #'TNG100-1-bh0' : ('galaxy-mergers_TNG100-1_gas-100_dm-100_star-100_bh-000.hdf5', fpath, 'b', 1.5),
    #    #'TNG100-1' : ('galaxy-mergers_TNG100-1_gas-100_dm-100_star-100_bh-001.hdf5', fpath, 
    #    #              np.array([100,100,0,0,100,1]).astype('int64'), 'b', 2.5),
    #    #'TNG100-2' : ('galaxy-mergers_TNG100-2_gas-012_dm-012_star-012_bh-001.hdf5', fpath, 'c', 1.5),
    #    #'TNG300-1-bh0' : ('galaxy-mergers_TNG300-1_gas-012_dm-012_star-012_bh-000.hdf5', fpath, 'm', 1.0),
    #    #'TNG300-1' : ('galaxy-mergers_TNG300-1_gas-012_dm-012_star-012_bh-001.hdf5', fpath, 
    #    #              np.array([12,12,0,0,12,1]).astype('int64'), 'm', 1.5),
    #    #'TNG300-1-N100' : ('galaxy-mergers_TNG300-1_gas-100_dm-100_star-100_bh-001.hdf5', fpath, 'pink', 1.5),
    #    #'TNG300-1-N100-bh0' : ('galaxy-mergers_TNG300-1_gas-100_dm-100_star-100_bh-000.hdf5', fpath, 'pink', 1)
    #    ##---'oldIll' : (None, 'brown', 2.5),
    #    ##---'Ill-1-nomprog' : ('galaxy_merger_files_with_no_mprog/galaxy-mergers_Illustris-1_gas-100_dm-100_star-100_bh-001.hdf5', 
    #    ##---                 ipath+'Illustris-1/output/', 'g', 2.5),
    #    ##---'TNG100-1-nomprog' : ('galaxy_merger_files_with_no_mprog/galaxy-mergers_TNG100-1_gas-100_dm-100_star-100_bh-001.hdf5', 
    #    ##---                      tpath+'TNG100-1/output/', 'b', 2.5),
    #    ##---'TNG100-1-bh0-nomprog' : ('galaxy_merger_files_with_no_mprog/galaxy-mergers_TNG100-1_gas-100_dm-100_star-100_bh-000.hdf5', 
    #    ##---                          tpath+'TNG100-1/output/', 'b', 1.5),
    #    ##---'TNG100-1-N012-bh0' : ('galaxy_merger_files_with_no_mprog/galaxy-mergers_TNG100-1_gas-012_dm-012_star-012_bh-000.hdf5', 
    #    ##---                       tpath+'TNG100-1/output/', 'darkblue', 2.5),
    #    ### dont use this file; it has at least one merger remnant with mbulge=0. prob need to rerun with Ngas=10
    #    ### ('galaxy-mergers_Illustris-1_gas-000_dm-010_star-010_bh-000.hdf5', 'darkgreen', 1.5)
    #}
    
    # ---- Loop thru dict and create dpops
    
    nfreqs = None
    
    for l in dpop_attrs.keys():
        if ('Illustris' in l) and (not inclIll): 
            continue
        if (l == 'oldIll') and (not inclOldIll):
            continue
        if ('TNG50' in l) and (not inclT50): 
            continue
        if ('TNG100' in l) and (not inclT100): 
            continue
        if ('TNG300' in l) and (not inclT300): 
                continue

        if not fsa_only:
            #if '-bh0' not in l:
            dp = Discrete(lbl=l, fixed_sepa=None, 
                          tau=tau, rchar=rchar, gamma_inner=gamma_inner, gamma_outer=gamma_outer, 
                          hard_model_type=hard_model_type,
                          nreals=nreals, nloudest=nloudest,
                          allow_mbh0=allow_mbh0, skip_evo=skip_evo, attrs=dpop_attrs[l],
                          subhalo_mstar_defn=subhalo_mstar_defn, bfrac=bfrac,
                          apply_mtot_mrat_cuts=apply_mtot_mrat_cuts)

            if nfreqs is None:
                nfreqs = dp.freqs.size
                print(f"NOTE: set {nfreqs=}")

            all_dpops = all_dpops + [dp]
            if 'Ill' not in l: 
                tng_dpops = tng_dpops + [dp]
            #else:
            #    print(f"Skipping run {l} with bh0")

            
        if fsa is not None:

            lbl='fsa-mm-'+l if mod_mmbulge else 'fsa-'+l
            dp_fsa = Discrete(lbl=lbl, fixed_sepa=fsa, 
                              tau=tau, rchar=rchar, gamma_inner=gamma_inner, gamma_outer=gamma_outer, 
                              hard_model_type=hard_model_type,
                              nreals=nreals, nloudest=nloudest, 
                              allow_mbh0=allow_mbh0, skip_evo=skip_evo, attrs=dpop_attrs[l], 
                              mod_mmbulge=mod_mmbulge, subhalo_mstar_defn=subhalo_mstar_defn, bfrac=bfrac, 
                              apply_mtot_mrat_cuts=apply_mtot_mrat_cuts)

            if nfreqs is None:
                nfreqs = dp_fsa.freqs.size
                print(f"NOTE: set {nfreqs=}")

            all_fsa_dpops = all_fsa_dpops + [dp_fsa]
            if 'Ill' not in l: 
                tng_fsa_dpops = tng_fsa_dpops + [dp_fsa]
            
            if ('TNG300' in l) and (inclT300) and (inclRescale):
                rescale_dp_fsa = Discrete(lbl='fsa-mm-r'+l, fixed_sepa=fsa, 
                                          tau=tau, rchar=rchar, gamma_inner=gamma_inner, gamma_outer=gamma_outer, 
                                          hard_model_type=hard_model_type,
                                          nreals=nreals, nloudest=nloudest, 
                                          allow_mbh0=allow_mbh0, skip_evo=skip_evo, 
                                          attrs=dpop_attrs[l], mod_mmbulge=True, 
                                          subhalo_mstar_defn=subhalo_mstar_defn, bfrac=bfrac, 
                                          rescale_mbulge=True, apply_mtot_mrat_cuts=apply_mtot_mrat_cuts)              
                all_fsa_dpops = all_fsa_dpops + [dp_fsa]
                tng_fsa_dpops = tng_fsa_dpops + [rescale_dp_fsa]

        print(f"{l} dpop_attrs: {dpop_attrs[l][0]} {dpop_attrs[l][1]} "
              f"{dpop_attrs[l][2]} {dpop_attrs[l][3]} {dpop_attrs[l][4]}")

    
    #if fsa is not None:
    data = all_dpops, tng_dpops, all_fsa_dpops, tng_fsa_dpops

    
    if pickle_dpops:
        print("\nSaving the dpops as a pkl file\n")
        
        if fsa is not None: 
            if fsa_only: 
                pickle_name += '_fsaOnly'
            else:
                pickle_name += '_withFsa'
        else:
            pickle_name += '_noFsaOrModMMb'
        
        if mod_mmbulge: 
            pickle_name += '_withModMMb'

        
        if (inclT300) and (inclRescale): pickle_name += 'withT300Rescale'
        
        pkl_fname = f"dpops_nfreqs{nfreqs}_nreals{nreals}_nloud{nloudest}_{pickle_name}.pkl"
        print(f"creating pkl file: {pkl_fname}")
        with open(f"{fpath}/{pkl_fname}", "wb") as f:
            pickle.dump(data, f)

    return data 

    #else:  
    #    return all_dpops, tng_dpops
    
    

if __name__ == '__main__':

    if len(sys.argv)>3:
        NREALS = int(sys.argv[1])
        NLOUD = int(sys.argv[2])
        SIMTYPE = sys.argv[3]
        TAU = float(sys.argv[4])
        if len(sys.argv)>5:
            print("Too many command line args ({sys.argv}).")
            sys.exit()

    else:
        print("expecting 4 command line args: NREALS, NLOUD, SIMTYPE, TAU.")
        sys.exit()
    
    inclIll=False
    inclT50=False
    inclT100=False
    inclT300=False
    if SIMTYPE == 'Illustris': inclIll = True
    if SIMTYPE == 'TNG50': inclT50 = True
    if SIMTYPE == 'TNG100': inclT100 = True
    if SIMTYPE == 'TNG300': inclT300 = True

    _HOME_PATH = Path('~/').expanduser()
    p = os.path.join(_HOME_PATH, 'nanograv/cosmo_sim_merger_data')
    if os.path.exists(p):
        _SIM_MERGER_PATH = p
    else:
        print("could not define _SIM_MERGER_PATH.")
        sys.exit()

    #for hardtype in ['old_rc10', 'old_rc100', 'old_rc10_nu0', 'ph15', 'ph15_rc10']:
    for hardtype in ['ph15', 'ph15_rc10']:
        for mmod in [True, False]:
            print(f"\n\ncreating dpops with {NREALS=}, {NLOUD=}, {SIMTYPE=}, {TAU=}, {hardtype=}, {mmod=}.")
            tmp = create_dpops(hard_model_type=hardtype, tau=TAU,
                               nreals=NREALS, nloudest=NLOUD, 
                               apply_mtot_mrat_cuts=True,allow_mbh0=True, 
                               bfrac=[0.5,1.0], skip_evo=False, 
                               subhalo_mstar_defn='MaxPastMass', fpath=_SIM_MERGER_PATH,
                               pickle_dpops=True,  
                               pickle_name=SIMTYPE, 
                               inclIll=inclIll, inclT50=inclT50, 
                               inclT100=inclT100, inclT300=inclT300,
                               fsa_only=True,
                               mod_mmbulge = mmod)
