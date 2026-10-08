"""Library Generation Script for the NG20 Parameter Spaces.

MPI-parallelized library generation for the NG20 parameter spaces
(``PS_NG20_Fiducial``, ``PS_NG20_Fiducial_Hard``, ``PS_NG20_Fiducial_Extended``), or for any other
parameter space specified by its class name.  Based on ``scripts/gen_lib_sams.py``, but using the
current :mod:`holodeck.librarian.gen_lib` machinery.

Usage
-----

mpirun -n <NPROCS> python ./scripts/gen_lib_ng20.py <PATH> -p <PSPACE> -n <SAMPS> -r <REALS> [-s <SHAPE>]

    <NPROCS> : number of processors to run on
    <PATH>   : output directory to save data to
    <PSPACE> : one of 'fiducial', 'hard', 'extended' (aliases for the NG20 parameter spaces), or the
               name of any parameter space class in `holodeck.librarian` (e.g. 'PS_Classic_Phenom').
               Default: 'fiducial'.
    <SAMPS>  : number of parameter-space samples for latin hyper-cube.  Default: 1e5
    <REALS>  : number of realizations at each parameter-space location.  Default: 1e3
    <SHAPE>  : SAM grid shape, either a single int (applied to all dimensions) or three ints giving
               the (M, q, z) shape.  Default: holodeck's default SAM grid.

Run ``python ./scripts/gen_lib_ng20.py -h`` for all options.

Examples:

    mpirun -n 64 python ./scripts/gen_lib_ng20.py output/ng20_fiducial -p fiducial -n 1e5 -r 1e3
    mpirun -n 64 python ./scripts/gen_lib_ng20.py output/ng20_ext -p extended -s 100 80 60

If a job is killed (e.g. by a wall-time limit), re-run the same command with ``--resume`` added: the
parameter space and configuration are loaded from the output directory, and existing simulation
files are skipped.

"""

__version__ = '0.1.0'

import argparse
import os
import shutil
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from mpi4py import MPI

import holodeck as holo
import holodeck.librarian
import holodeck.librarian.combine
from holodeck import log
from holodeck.librarian import gen_lib


comm = MPI.COMM_WORLD

#: aliases for the parameter spaces tested in `notebooks/devs/ng20_all_param_spaces.ipynb`
PSPACE_ALIASES = {
    'fiducial': 'PS_NG20_Fiducial',
    'hard': 'PS_NG20_Fiducial_Hard',
    'extended': 'PS_NG20_Fiducial_Extended',
}


def main():

    # ---- setup arguments, loggers, and outputs

    if comm.rank == 0:
        # catch exits (e.g. `-h` or bad arguments) so that they can be shared with all processes
        try:
            args = _setup_argparse()
        except SystemExit as err:
            args = err
    else:
        args = None

    args = comm.bcast(args, root=0)
    if isinstance(args, SystemExit):
        sys.exit(args.code)

    gen_lib._setup_log(comm, args)

    if comm.rank == 0:
        # get parameter-space (created new, or loaded from previous save when `args.resume`)
        space = gen_lib._setup_param_space(args)

        if args.resume:
            max_failures = args.max_failures
            args, config_fname = gen_lib.load_config_from_path(args.output, log)
            log.warning(f"Loaded configuration save from {config_fname}")
            # `args.resume` may be set to `False` after loading from save; reset to True
            args.resume = True
            args.max_failures = max_failures
        else:
            space_fname = space.save(args.output)
            log.info(f"Saved parameter space {space} to {space_fname}")
            config_fname = gen_lib._save_config(args)
            log.info(f"Saved configuration to {config_fname}")
            dst_file = args.output.joinpath("runtime_" + Path(__file__).name)
            shutil.copyfile(__file__, dst_file)
            log.info(f"Copied {__file__} to {dst_file}")

        # Split and distribute index numbers to all processes
        npars = args.nsamples
        indices = range(npars)
        indices = np.random.permutation(indices)
        indices = np.array_split(indices, comm.size)
        num_ind_per_proc = [len(ii) for ii in indices]
        log.warning(
            f"param_space={args.param_space}, parameters={space.nparameters}, samples={npars}, "
            f"sam_shape={args.sam_shape}, nreals={args.nreals}, nfreqs={args.nfreqs}, "
            f"pta_dur={args.pta_dur} [yr] || cores={comm.size}, "
            f"max runs per core = {np.max(num_ind_per_proc)}"
        )
    else:
        space = None
        indices = None

    space = comm.bcast(space, root=0)
    args = comm.bcast(args, root=0)
    indices = comm.scatter(indices, root=0)

    iterator = holo.utils.tqdm(indices) if (comm.rank == 0) else np.atleast_1d(indices)

    comm.barrier()
    beg = datetime.now()
    log.info(f"beginning tasks at {beg}")
    failures = 0

    for par_num in iterator:
        log.debug(f"{comm.rank=} {par_num=}")
        params = space.param_dict(par_num)

        rv, _sim_fname = gen_lib.run_sam_at_pspace_params(args, space, par_num, params)
        if rv is False:
            failures += 1

        if (args.max_failures is not None) and (failures > args.max_failures):
            err = f"Failed {failures} times on rank:{comm.rank}!"
            log.exception(err)
            raise RuntimeError(err)

    end = datetime.now()
    dur = (end - beg)
    log.info(f"\t{comm.rank} done at {str(end)} after {str(dur)} = {dur.total_seconds()} ({failures=})")

    # Make sure all processes are done so that all files are ready for merging
    comm.barrier()

    if (comm.rank == 0):
        log.warning("Combining simulation files into single library file")
        holo.librarian.combine.sam_lib_combine(args.output, log)
        log.warning("Library combination completed.")

    return


def _int_from_float(val):
    """Convert strings like '1e5' to integers."""
    fval = float(val)
    ival = int(fval)
    if ival != fval:
        raise argparse.ArgumentTypeError(f"'{val}' is not an integer!")
    return ival


def _setup_argparse():
    """Parse command-line arguments, then construct the `args` used by `holodeck.librarian.gen_lib`.
    """
    lib = holo.librarian
    aliases = ", ".join(f"'{kk}' ({vv})" for kk, vv in PSPACE_ALIASES.items())

    parser = argparse.ArgumentParser(
        description="Generate a holodeck SAM library (MPI parallelized).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('output', type=str, help='output path [created if doesnt exist]')
    parser.add_argument('-p', '--pspace', type=str, default='fiducial',
                        help=f"parameter space: one of {aliases}, or a parameter-space class name")

    parser.add_argument('-n', '--nsamples', type=_int_from_float, default=100_000,
                        help='number of parameter space samples')
    parser.add_argument('-r', '--nreals', type=_int_from_float, default=1_000,
                        help='number of realizations at each parameter-space sample')
    parser.add_argument('-s', '--sam_shape', type=int, nargs='+', default=None,
                        help="SAM grid shape: one int for all dimensions, or three ints for (M, q, z). "
                        "Default (None) uses holodeck's default SAM grid")
    parser.add_argument('-f', '--nfreqs', type=int, default=lib.DEF_NUM_FBINS,
                        help='number of frequency bins')
    parser.add_argument('-d', '--dur', type=float, default=lib.DEF_PTA_DUR,
                        help='PTA observing duration [yrs]')
    parser.add_argument('-l', '--nloudest', type=int, default=lib.DEF_NUM_LOUDEST,
                        help='number of loudest single sources')

    parser.add_argument('--gwb', default=True, action=argparse.BooleanOptionalAction,
                        help="calculate and store the 'gwb' per se")
    parser.add_argument('--ss', default=True, action=argparse.BooleanOptionalAction,
                        help="calculate and store SS/CW sources and the BG separately")
    parser.add_argument('--params', default=True, action=argparse.BooleanOptionalAction,
                        help="calculate and store SS/BG binary parameters [NOTE: requires `--ss`]")

    parser.add_argument('--resume', action='store_true', default=False,
                        help='resume a library by loading the parameter-space and config from `output`')
    parser.add_argument('--recreate', action='store_true', default=False,
                        help='recreate existing simulation files')
    parser.add_argument('--seed', type=int, default=None, help='random seed to use')
    parser.add_argument('--max-failures', dest='max_failures', type=int, default=None,
                        help='maximum number of failed simulations per process before aborting (None: no limit)')
    parser.add_argument('-v', '--verbose', metavar='LEVEL', type=int, nargs='?', const=20, default=30,
                        help='verbose output level (DEBUG=10, INFO=20, WARNING=30)')

    args = parser.parse_args()

    # ---- resolve and check parameter-space name

    param_space = PSPACE_ALIASES.get(args.pspace.lower(), args.pspace)
    if ("." not in param_space) and (param_space not in lib.param_spaces_dict):
        parser.error(
            f"Unknown parameter space '{args.pspace}'!  Use one of {aliases}, "
            f"or one of: {', '.join(lib.param_spaces_dict.keys())}"
        )

    # ---- check sam_shape

    sam_shape = args.sam_shape
    if sam_shape is not None:
        if len(sam_shape) == 1:
            sam_shape = sam_shape[0]
        elif len(sam_shape) == 3:
            sam_shape = tuple(sam_shape)
        else:
            parser.error(f"`--sam_shape` must be given one or three integers, not {args.sam_shape}!")

    # ---- construct `args` using `gen_lib`, which sets up the output directories

    argv = [
        param_space, args.output,
        '-n', str(args.nsamples), '-r', str(args.nreals), '-f', str(args.nfreqs),
        '-d', str(args.dur), '-l', str(args.nloudest), '-v', str(args.verbose),
        '--gwb' if args.gwb else '--no-gwb',
        '--ss' if args.ss else '--no-ss',
        '--params' if args.params else '--no-params',
    ]
    if args.seed is not None:
        argv += ['--seed', str(args.seed)]
    if args.resume:
        argv.append('--resume')
    if args.recreate:
        argv.append('--recreate')

    # NOTE: `sam_shape` is set afterwards, as `gen_lib` only accepts a single integer
    args = gen_lib._setup_argparse(argv)
    args.sam_shape = sam_shape
    args.max_failures = args.max_failures

    return args


def mpiabort_excepthook(type, value, traceback):
    sys.__excepthook__(type, value, traceback)
    comm.Abort()
    return


if __name__ == "__main__":
    np.seterr(divide='ignore', invalid='ignore', over='ignore')
    warnings.filterwarnings("ignore", category=UserWarning)
    sys.excepthook = mpiabort_excepthook
    beg_time = datetime.now()
    beg_time = comm.bcast(beg_time, root=0)

    if comm.rank == 0:
        this_fname = os.path.abspath(__file__)
        head = f"holodeck :: {this_fname} : {str(beg_time)} - rank: {comm.rank}/{comm.size}"
        head = "\n" + head + "\n" + "=" * len(head) + "\n"
        print(head, flush=True)

    main()

    if comm.rank == 0:
        end = datetime.now()
        dur = end - beg_time
        tail = f"Done at {str(end)} after {str(dur)} = {dur.total_seconds()}"
        print("\n" + "=" * len(tail) + "\n" + tail + "\n", flush=True)

    sys.exit(0)
