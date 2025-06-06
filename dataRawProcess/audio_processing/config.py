import json
import argparse

def load_cfg(config_path):
    """Load configuration from JSON file"""
    with open(config_path) as f:
        return json.load(f)

def parse_args(args_list=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/home4/tuanlha/DataTest", help="input folder path")
    parser.add_argument("--playlist_name", type=str, default="", help="input folder path")
    parser.add_argument("--config_path", type=str, default="./config.json", help="path to config file")
    parser.add_argument("--cuda_id", type=int, default=0, help="CUDA device ID")
    parser.add_argument("--label_audio", type=bool, default=False, help="Label audio files")
    parser.add_argument("--remove_minority", type=bool, default=False, help="Remove minority speakers in diarization")

    if args_list is not None:
        return parser.parse_args(args_list)
    
    return parser.parse_args()