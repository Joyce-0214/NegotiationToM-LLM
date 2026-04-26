from argparse import ArgumentParser
from craigslistbargain.options import add_model_arguments

parser = ArgumentParser()
add_model_arguments(parser)
opt = parser.parse_args([])

print(opt.tom_model)
print(opt.sa_d_hist, opt.sa_d_ctx, opt.sa_d_buyer)
print(opt.sa_d_obs, opt.sa_d_h, opt.sa_k_type, opt.sa_d_core, opt.sa_f_num)