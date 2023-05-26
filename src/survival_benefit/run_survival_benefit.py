########## This is ROW APPROACH ############
import argparse
import numpy as np
from survival_benefit.survival_benefit_class import SurvivalBenefit

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate possible distributions of survival benefit",
        epilog='Implemented by Haeun Hwangbo')

    parser.add_argument('--indir', type=str,
                        help='Input data directory')
    parser.add_argument('--mono', type=str,
                        help='Monotherapy data filename prefix')
    parser.add_argument('--comb', type=str,
                        help='Combination therapy data filename prefix')
    parser.add_argument('--outdir', type=str,
                        help='Output directory')
    parser.add_argument('--out-prefix', type=str, default=None,
                        help='Output filename prefix (default: name of combination therapy)')
    parser.add_argument('--N', type=int, default=1000,
                        help='Patient number to populate')
    parser.add_argument('--figsize', nargs=2, type=int, default=[6, 4],
                        help='Output figure width and height (default: (6, 4))')
    parser.add_argument('--fig-format', default='png', choices=['pdf', 'png', 'jpg'],
                        help='File extension for output figure (default: pdf)')
    args = parser.parse_args()
    return args



#for coef in [0]:
#for coef in np.power(1.1, np.arange(0, 24, 8)):
#for coef in [-1.9, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 1.9]:
#for coef in [-50, -25, -10, -5, -2, -1.5, -1, -0.5, 0, 1]:
for coef in [-50, -25, -10, -5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 5, 10]:
    sb.fit_curve(prob_kind='power', prob_coef=coef, prob_offset=0)
    sb.plot_fit_curve_sanity_check()
    sb.plot_benefit_distribution(kind='absolute')
    sb.plot_benefit_distribution(kind='ratio')
    sb.plot_t_delta_t_corr()
    sb.save_record()
    sb.save_summary_stats()
