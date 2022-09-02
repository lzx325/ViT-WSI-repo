import sys
import yaml
from pprint import pprint
import argparse

def parse_args(parser=None):
    if parser is None:
        parser=argparse.ArgumentParser(allow_abbrev=False)
        parser.set_defaults(func=None)
        parser.add_argument("mode",type=str)
        parser.add_argument("--config",type=str)
    args=parser.parse_args()
    json_fp=args.config
    with open(json_fp) as f:
        d=yaml.safe_load(f)
    for k,v in d.items():
        # if isinstance(v,dict):
        #     sub_args=argparse.Namespace()
        #     sub_args.__dict__.update(v)
        #     setattr(args,k,sub_args)
        # else:
            setattr(args,k,v)
    return args
