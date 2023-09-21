# @Time   : 2020/7/20
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE
# @Time   : 2022/7/8, 2020/10/3, 2020/10/1
# @Author : Zhen Tian, Yupeng Hou, Zihan Lin
# @Email  : chenyuwuxinn@gmail.com, houyupeng@ruc.edu.cn, zhlin@ruc.edu.cn

import argparse
import os
import time
from ast import arg

import pandas as pd

from recbole.quick_start import run_recbole, run_recboles
from recbole.utils import list_to_latex
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str,
                        default="BPR", help="name of models")
    parser.add_argument(
        "--dataset", "-d", type=str, default="ml-100k", help="name of datasets"
    )
    parser.add_argument("--config_files", type=str,
                        default=None, help="config files")
    parser.add_argument(
        "--nproc", type=int, default=1, help="the number of process in this group"
    )
    parser.add_argument(
        "--ip", type=str, default="localhost", help="the ip of master node"
    )
    parser.add_argument(
        "--port", type=str, default="5678", help="the port of master node"
    )
    parser.add_argument(
        "--world_size", type=int, default=-1, help="total number of jobs"
    )
    parser.add_argument(
        "--group_offset",
        type=int,
        default=0,
        help="the global rank offset of this group",
    )

    args, _ = parser.parse_known_args()
    now = time.strftime("%y%m%d%H%M%S")

    config_file_list = (
        args.config_files.strip().split(" ") if args.config_files else None
    )
    config_dict = {'now': now}

    start = time.time()
    nproc = torch.cuda.device_count() if args.nproc == -1 else args.nproc

    if nproc == 1 and args.world_size <= 0:

        run_recbole(model=args.model, dataset=args.dataset,
                             config_file_list=config_file_list, config_dict=config_dict)
    else:
        if args.world_size == -1:
            args.world_size = nproc
        import torch.multiprocessing as mp
        mp.spawn(
            run_recboles,
            args=(
                args.model,
                args.dataset,
                config_file_list,
                args.ip,
                args.port,
                args.world_size,
                nproc,
                args.group_offset,
            ),
            nprocs=nproc,
        )
    elapse = (time.time() - start) / 60  # unit: mins
    print(f'Elapse: {elapse:.2f} mins')

    # https://pytorch.org/tutorials/advanced/generic_join.html#distributed-training-with-uneven-inputs-using-the-join-context-manager