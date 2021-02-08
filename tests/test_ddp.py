import json
import os
import sys
from typing import Dict

import pytest
import torch

from test_numerical import test_fmoe as _test_fmoe
from test_numerical import test_fmoe_linear as _test_fmoe_linear


def _run_distributed(func, args: Dict):
    import subprocess
    import os

    ps, n = [], 2
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "36666"
    os.environ["OMPI_COMM_WORLD_SIZE"] = str(n)

    for i in range(n):
        os.environ["OMPI_COMM_WORLD_RANK"] = str(i)
        p = subprocess.Popen(
            [sys.executable, __file__, func, json.dumps(args)], stdout=subprocess.PIPE,
        )
        ps.append(p)

    for p in ps:
        p.wait()
        retc = p.poll()
        assert retc == 0


@pytest.mark.parametrize("num_expert", [4, 8])
@pytest.mark.parametrize("top_k", [2])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("d_model", [16])
@pytest.mark.parametrize("d_hidden", [32])
def test_fmoe_linear_distributed(
    num_expert, top_k, batch_size, d_model, d_hidden,
):
    _run_distributed(
        "_test_fmoe_linear",
        {
            "num_expert": num_expert,
            "top_k": top_k,
            "batch_size": batch_size,
            "d_model": d_model,
            "d_hidden": d_hidden,
        },
    )


@pytest.mark.parametrize("num_expert", [4, 8])
@pytest.mark.parametrize("top_k", [2])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("d_model", [16])
@pytest.mark.parametrize("expert", ["NaiveExpert", "LinearExpert"])
def test_fmoe_distributed(
    num_expert, top_k, batch_size, d_model, expert,
):
    _run_distributed(
        "_test_fmoe",
        {
            "num_expert": num_expert,
            "top_k": top_k,
            "batch_size": batch_size,
            "d_model": d_model,
            "expert": expert,
        },
    )


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        args = json.loads(sys.argv[2])
        os.environ["RANK"] = os.environ.get("OMPI_COMM_WORLD_RANK", "0")
        os.environ["WORLD_SIZE"] = os.environ.get("OMPI_COMM_WORLD_SIZE", "1")
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["RANK"]
        torch.distributed.init_process_group(backend="nccl")
        args["rank"] = torch.distributed.get_rank()
        args["world_size"] = torch.distributed.get_world_size()
        locals()[sys.argv[1]](**args)
