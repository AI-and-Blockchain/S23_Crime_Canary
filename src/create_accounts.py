import csv, os, pandas as pd
from algosdk import account, mnemonic

f_object = 'accounts.csv'


if not(os.path.exists(f_object)):
    policeWallet = account.generate_account()
    OracleCreator = account.generate_account()
    participant = account.generate_account()
    
    pol_secret = mnemonic.from_private_key(policeWallet[0])
    orac_secret = mnemonic.from_private_key(OracleCreator[0])
    par_secret = mnemonic.from_private_key(participant[0])
    
    with open(f_object, 'w', newline='') as f_object:
        writer = csv.writer(f_object)
        
        writer.writerow(["ID", "TYPE", "PUBLIC", "PRIVATE", "SECRET"])
        writer.writerow([1, "POLICE", policeWallet[1], policeWallet[0], pol_secret])
        writer.writerow([2, "ORACLE", OracleCreator[1], OracleCreator[0], orac_secret])
        writer.writerow([3, "USER", participant[1], participant[0], par_secret])

else:    
    no_lines = len(pd.read_csv(f_object))
    with open(f_object, 'a') as f_object:
        writer = csv.writer(f_object)
        participant = account.generate_account()
        par_secret = mnemonic.from_private_key(participant[0])
        writer.writerow([no_lines + 1, "USER", participant[1], participant[0], par_secret])
