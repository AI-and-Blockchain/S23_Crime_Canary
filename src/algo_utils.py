import base64, numpy as np
from algosdk.v2client.algod import AlgodClient
from algosdk.v2client.indexer import IndexerClient


def decode_message(hash: str) -> str:
    return base64.b64decode(hash).decode('utf-8')


def verify_transaction(img_hash: str, pkey: str, indexer: IndexerClient) -> bool:
    try:
        response = indexer.search_transactions_by_address(address=pkey, txn_type="pay", note_prefix=str.encode(img_hash))
        txn_hash = decode_message(response['transaction']['note'])
        if img_hash == txn_hash:
            return True 
    except:
        return False 
    return False


def get_testnet_client() -> AlgodClient:
    algod_address = "http://localhost:4001"
    algod_token = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    return AlgodClient(algod_token, algod_address) 


def get_testnet_indexer() -> IndexerClient:
    indexer_addr = "https://algoindexer.testnet.algoexplorerapi.io"
    indexer_token = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    return IndexerClient(indexer_token, indexer_addr)


def hash_image(img: np.ndarray, hasher) -> str:
    val = ""
    for num in hasher.compute(img).ravel().astype('str'):
        val += num 
    return val