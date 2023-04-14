
from algosdk.v2client import algod
from algosdk import transaction

#Client declaration function
def client_transaction(API_address, token):
    algod_address = API_address
    algod_token = token
    headers = {"X-API-Key": algod_token,}
    algod_clien = algod.AlgodClient(algod_token, algod_address, headers)
    return algod_clien

#Transaction details
API_address = "https://testnet-algorand.api.purestake.io/ps2"
token = "3UeCpd3CVz9lvWElfm0sr3zQzvP6FyOC136j7WvQ "
algod_client = client_transaction(API_address, token)
account = "2MAAWMCSMYROAA5OHMZSJM7AIURXVRCXZCCOFPERXWODVQMGF33IX7BB4A"
privat_key = "cPsho67oRHxhVDJWcHuyz+ZtC143072bU9PcQlXdLH/TAAswUmYi4AOuOzMks+BFI3rEV8iE4ryRvZw6wYYu9g=="
parameters = algod_client.suggested_params()
asset_id = 176869006   #Canary token ID



#Opt in canary token

txn = transaction.AssetTransferTxn(sender=account,sp=parameters,receiver=account,amt=0,index=asset_id)
signed_txn = txn.sign(privat_key)
txid = algod_client.send_transaction(signed_txn)
print("opt-in txid is: ", txid)
transaction.wait_for_confirmation(algod_client, txid)




















