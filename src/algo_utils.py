import json
import hashlib
import base64, numpy as np
import os, pandas as pd
from algosdk import transaction
from algosdk.v2client.algod import AlgodClient
from algosdk.v2client.indexer import IndexerClient

DEFAULT_ACCOUNT_FILE = '../accounts.csv'

def get_default_wallet():
    df = pd.read_csv(DEFAULT_ACCOUNT_FILE).iloc[0]
    return df['PUBLIC'], df['PRIVATE']
    

def decode_message(hash: str) -> str:
    return base64.b64decode(hash).decode('utf-8')


def verify_transaction(img_hash: str, pkey: str, indexer: IndexerClient) -> bool:
    try:
        response = indexer.search_transactions_by_address(address=pkey, txn_type="pay", note_prefix=str.encode(img_hash))
        txn_hash = decode_message(response['transactions'][0]['note'])
    except:
        return False 
    else:
        if img_hash == txn_hash:
            return True 


def get_testnet_client() -> AlgodClient:
    algod_address = "http://localhost:4001"
    algod_token = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    return AlgodClient(algod_token, algod_address) 


def get_testnet_indexer() -> IndexerClient:
    indexer_addr = "https://algoindexer.testnet.algoexplorerapi.io"
    indexer_token = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    return IndexerClient(indexer_token, indexer_addr)


def hash_image(img: np.ndarray) -> str:
    b = hashlib.sha256(img).hexdigest()
    return b


def print_asset_holding(client: AlgodClient, account: str, assetid: str):
    # note: if you have an indexer instance available it is easier to just use this
    # response = myindexer.accounts(asset_id = assetid)
    # then loop thru the accounts returned and match the account you are looking for
    account_info = client.account_info(account)
    idx = 0
    for my_account_info in account_info['assets']:
        scrutinized_asset = account_info['assets'][idx]
        idx = idx + 1        
        if (scrutinized_asset['asset-id'] == assetid):
            print("Asset ID: {}".format(scrutinized_asset['asset-id']))
            print(json.dumps(scrutinized_asset, indent=4))
            break
        

# utility for waiting on a transaction confirmation
def wait_for_confirmation(client: AlgodClient, transaction_id, timeout: int):
    """
    Wait until the transaction is confirmed or rejected, or until 'timeout'
    number of rounds have passed.
    Args:
        transaction_id (str): the transaction to wait for
        timeout (int): maximum number of rounds to wait    
    Returns:
        dict: pending transaction information, or throws an error if the transaction
            is not confirmed or rejected in the next timeout rounds
    """
    start_round = client.status()["last-round"] + 1;
    current_round = start_round

    while current_round < start_round + timeout:
        try:
            pending_txn = client.pending_transaction_info(transaction_id)
        except Exception:
            return 
        if pending_txn.get("confirmed-round", 0) > 0:
            return pending_txn
        elif pending_txn["pool-error"]:  
            raise Exception(
                'pool error: {}'.format(pending_txn["pool-error"]))
        client.status_after_block(current_round)                   
        current_round += 1
    raise Exception(
        'pending tx not found in timeout rounds, timeout value = : {}'.format(timeout))


def send_transaction(client: AlgodClient, stxn):
    try:
        txid = client.send_transaction(stxn)
        print("Signed transaction with txID:", txid)
        confirmation = wait_for_confirmation(client, txid, timeout=10)
        print("Confirmed in round:", confirmation['confirmed-round'])
    except Exception as error:
        print(error)
        return False
    else:
        return True
        

def make_send_transaction(client: AlgodClient, sender: str, receiver: str, assetid: str, sender_skey: str, send_amount: int, note: str='') -> bool:
    params = client.suggested_params()
    
    asset_tf = transaction.AssetTransferTxn(
        sender=sender,
        sp=params,
        receiver=receiver,
        amt=send_amount,
        index=assetid,
        note=note
    )
    
    signed_asset_tf = asset_tf.sign(sender_skey)
    return send_transaction(client, signed_asset_tf)