import argparse

import yaml

from preprocessor.preprocessor import Preprocessor
from preprocessor import mihoyo, haveasip, visec, lmh_final, esd, tth_final


def main(config):
    # if "LJSpeech" in config["dataset"]:
    #     ljspeech.prepare_align(config)
    # if "VCTK" in config["dataset"]:
    #     vctk.prepare_align(config)
    # if "ESD" in config["dataset"]:
    #     esd.prepare_align(config)
    #     esd.make_meta_dict(config)
    if "Mihoyo" in config["dataset"]:
        mihoyo.prepare_align(config)
    # if "HaveASip" in config["dataset"]:
    #     haveasip.prepare_align(config)
    # if "Visec" in config["dataset"]:
    #     visec.prepare_align(config)
    # if "LMH_final" in config["dataset"]:
    #     lmh_final.prepare_align(config)
    # if "TTH_final" in config["dataset"]:
    #     tth_final.prepare_align(config)

    preprocessor = Preprocessor(config)
    preprocessor.build_from_path()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/ESD/preprocess.yaml")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    
    main(config)
    
    
