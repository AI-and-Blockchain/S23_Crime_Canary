import csv, os, pandas as pd
from algosdk.v2client import algod
from algosdk import transaction

#Client declaration function
def client_transaction(API_address, token):
    algod_address = API_address
    algod_token = token
    headers = {"X-API-Key": algod_token,}
    algod_clien = algod.AlgodClient(algod_token, algod_address, headers)
    return algod_clien

f_object = 'accounts.csv'
c_object = 'token.csv'

#Retrieve Police public and private key
with open(f_object, 'r') as csv_file:
    csv_reader = list(csv.reader(csv_file))
    private_key = csv_reader[1][3]
    token_manager = csv_reader[1][2]

#Transaction details
API_address = "https://testnet-algorand.api.purestake.io/ps2"
token = "3UeCpd3CVz9lvWElfm0sr3zQzvP6FyOC136j7WvQ"
algod_client = client_transaction(API_address, token)
parameters = algod_client.suggested_params()
parameters.fee = 1000
parameters.flat_fee = True

if not(os.path.exists(c_object)):
    # Create Asset
    token_name = "canary"
    unit_name = token_name.upper()
    token_count = 1000
    unsigned_transac = transaction.AssetConfigTxn(sender=token_manager, sp=parameters, total=token_count, default_frozen=False,
                                                  unit_name=unit_name, asset_name=token_name, manager=token_manager,
                                                  reserve=token_manager, freeze=token_manager, clawback=token_manager,
                                                  url="https://path/to/my/asset/details", decimals=0)
    # Sign Asset transaction
    signed_transac = unsigned_transac.sign(private_key)

    # Send the transaction to the network and retrieve the txid.
    txid = algod_client.send_transaction(signed_transac)

    #Retrieve the asset ID of the newly created asset by first ensuring that the creation transaction was confirmed,
    #then grabbing the asset id from the transaction.Wait for the transaction to be confirmed
    #transaction.wait_for_confirmation(algod_client, txid)
    try:
        ptx = algod_client.pending_transaction_info(txid)
        asset_id = ptx["asset-index"]
        with open(c_object, 'w', newline='') as c_object:
            writer = csv.writer(c_object)
            writer.writerow(["ASSET","UNIT_NAME","COUNT","MANAGER","TX_ID", "ASSET_ID"])
            writer.writerow([token_name, unit_name, token_count, token_manager, txid, asset_id])

    except Exception as e:
        print(e)












