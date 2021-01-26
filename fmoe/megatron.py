from torch import nn
from .moe import FFFN


def create_moe_mlp(args):
    assert args.num_experts % args.model_parallel_size == 0, 'Num experts should be multiple of mp size'
    num_experts = args.num_experts // args.model_parallel_size 
    fmoe = FFFN(num_experts, 
            d_model=args.hidden_size, 
            d_hidden=args.hidden_size * 4,
            world_size=args.model_parallel_size)
    return fmoe
    
