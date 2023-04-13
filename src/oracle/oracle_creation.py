from pyteal import *
from oracle_utils import create_deploy_oracle
import csv, os, pandas as pd
from algosdk import account, mnemonic

#Generate account for the Oracle owner
OracleCreator = account.generate_account()
orac_secret = mnemonic.from_private_key(OracleCreator[0])

with open('oracle_owner.csv', 'w', newline='') as f_object:
    writer = csv.writer(f_object)
    writer.writerow(["ID", "TYPE", "PUBLIC", "PRIVATE", "SECRET"])
    writer.writerow([1, "ORACLE", OracleCreator[1], OracleCreator[0], orac_secret])



app_id = create_deploy_oracle(OracleCreator[1], OracleCreator[0])
print("Created application ID is: ", app_id)

#Save the deployed application ID
with open('oracle_contract.csv', 'w', newline='') as csvfile:
    my_writer = csv.writer(csvfile)
    my_writer.writerow(["APP_ID"])
    my_writer.writerow([app_id])


