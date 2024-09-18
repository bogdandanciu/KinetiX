#!/usr/bin/env python3

# Standard library imports
import os

# Standard library imports
import sys

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

try:
    # Try importing as if it's part of the installed package
    from generator.core.generate import generate_files
    from generator.utils import general_utils as gutils
except ImportError:
    # If that fails, try importing as if it's a local script
    from core.generate import generate_files
    from utils import general_utils as gutils


def main(args=None):
    if args is None:
        args = gutils.get_parser()
        generate_files(mech_file=args.mechanism,
                       output_dir=args.output,
                       single_precision=args.single_precision,
                       header_only=args.header_only,
                       unroll_loops=args.unroll_loops,
                       align_width=args.align_width,
                       target=args.target,
                       loop_gibbsexp=args.loop_gibbsexp,
                       group_rxnunroll=args.group_rxnunroll,
                       transport=args.transport,
                       group_vis=args.group_vis,
                       nonsymDij=args.nonsymDij,
                       rcp_diffcoeffs=args.fit_rcpdiffcoeffs
                       )


if __name__ == '__main__':
    sys.exit(main())
