import sys, os
import argparse
import numpy as np
from PIL import Image
from algosdk import transaction
from algo_utils import get_default_wallet, get_testnet_client, make_send_transaction, hash_image, send_transaction

CLIENT = get_testnet_client()

DEFAULT_WALLET_PKEY, _ = get_default_wallet()
DEFAULT_ASSET_ID = 176869006

STAKE_FEE = 1

def create_arg_parser():
    parser = argparse.ArgumentParser(description='Description of your app.')
    
    parser.add_argument('pkey', help='Public key.')
    
    parser.add_argument('skey', help='Secret key.')
    
    parser.add_argument('imagePath',
                    help='Path to image.')
    return parser

if __name__ == "__main__":
    arg_parser = create_arg_parser()
    parsed_args = arg_parser.parse_args(sys.argv[1:])
    
    user_pkey, user_skey, path = parsed_args.pkey, parsed_args.skey, parsed_args.imagePath
    
    if os.path.exists(path):
        image = Image.open(path)
        image.load()
        image = np.asarray(image)
        
        print("OPTING IN TO ASSET {0} FOR {1}".format(DEFAULT_ASSET_ID, user_pkey))
        # OPT IN
        make_send_transaction(CLIENT, user_pkey, user_pkey, DEFAULT_ASSET_ID, user_skey, 0)    
        
        print("STAKING")
        # STAKE
        stake = transaction.PaymentTxn(user_pkey, CLIENT.suggested_params(), DEFAULT_WALLET_PKEY, amt=STAKE_FEE, note=hash_image(image).encode())
        signed_stake = stake.sign(user_skey)
        send_transaction(CLIENT, signed_stake)
        
        print("SENT IMAGE HASH: {0}".format(hash_image(image)))