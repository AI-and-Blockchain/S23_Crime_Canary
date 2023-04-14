
import csv
from oracle_utils import opt_in_app
from algosdk.v2client import algod

# initialize an algodClient
#algod_token = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
#algod_address = "http://localhost:4001"
#algod_client = algod.AlgodClient(algod_token, algod_address)
algod_address = "https://testnet-algorand.api.purestake.io/ps2"
algod_token = "3UeCpd3CVz9lvWElfm0sr3zQzvP6FyOC136j7WvQ"
headers = {"X-API-Key": algod_token, }
algod_client = algod.AlgodClient(algod_token, algod_address, headers)

f_object = '../accounts.csv'
c_object = '../oracle_contract.csv'

#Retrieve Oracle application ID
with open(c_object, 'r') as csv_file:
    csv_reader = list(csv.reader(csv_file))
    app_id = csv_reader[1][0]

#Retrieve Police public and private key
with open(f_object, 'r') as csv_file:
    csv_reader = list(csv.reader(csv_file))
    police_private_key = csv_reader[1][3]
    police_public_key = csv_reader[1][2]
    participant_address = csv_reader[2][2]
    participant_private_key = csv_reader[2][3]
    participant2_address = csv_reader[3][2]
    participant2_private_key = csv_reader[3][3]

#Participant opt-in to application
opt_in_app(algod_client, participant_address, participant_private_key, app_id)
#Participant2 opt-in to application
opt_in_app(algod_client, participant2_address, participant2_private_key, app_id)
#Police opt-in to application
opt_in_app(algod_client, police_public_key, police_private_key, app_id)

'''How to call application'''
#app_args = [b"notify_classification", intToBytes(prediction), intToBytes(severity)]
#note = b"classification-notification"
#call_app(algod_client, police_public_key, police_private_key, app_id, app_args, [participant_address], note)
