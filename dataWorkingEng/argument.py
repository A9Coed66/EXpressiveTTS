import argparse

def get_config():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('action', type=str, default='train', help='Action') # train / test / evaluate
    parser.add_argument('--config', default='./config/Mihoyo/base.yaml', help='config yaml file')
    parser.add_argument('--num_worker', type=int, default=0, help='Num workers')
    parser.add_argument('--seed', type=int, default=100, help='seed number')
    parser.add_argument('--n_timesteps', type=int, default=50, help='T')
    parser.add_argument('--device', type=str, default='cuda:0', help='Cuda device')
    parser.add_argument('--logging', type=bool, default=False, help='Logging option')
    parser.add_argument('--resume', type=str, default=None, help='Resume option')
    parser.add_argument('--tag', type=str, default=None, help='Tagging')
    parser.add_argument('--pa', type=bool, default=True, help='parallel sentence')
    parser.add_argument('--env', type=str, default='local', help='Enviornment')
    parser.add_argument('--test_checkpoint', type=str, default='test', help='Exp number')
    parser.add_argument('--test_file', type=str, default='./test_sentence/mihoyo_sentence.txt', help='path to a file with texts to synthesize')
    parser.add_argument('--d_control', type=float, default=1.0, help='Control the duration')
    parser.add_argument('--p_control', type=float, default=1.0, help='Control the pitch')
    parser.add_argument('--e_control', type=float, default=1.0, help='Control the energy')
    arguments = parser.parse_args()
    
    return arguments