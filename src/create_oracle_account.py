
import csv, os
from algosdk import account, mnemonic

#Generate account for the Oracle owner
OracleCreator = account.generate_account()
orac_secret = mnemonic.from_private_key(OracleCreator[0])

f_object = 'oracle_owner.csv'
if not(os.path.exists(f_object)):
    with open(f_object, 'w', newline='') as f_object:
        writer = csv.writer(f_object)
        writer.writerow(["ID", "TYPE", "PUBLIC", "PRIVATE", "SECRET"])
        writer.writerow([1, "ORACLE", OracleCreator[1], OracleCreator[0], orac_secret])


print("Oracle owner's public key is: ", OracleCreator[1])