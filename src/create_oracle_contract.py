
from oracle_utils import create_deploy_oracle
import csv, os


c_object = 'oracle_owner.csv'

#Retrieve Oracle owner details
with open(c_object, 'r') as csv_file:
    csv_reader = list(csv.reader(csv_file))
    public_key = csv_reader[1][2]
    private_key = csv_reader[1][3]


f_object = 'oracle_contract.csv'
if not(os.path.exists(f_object)):
    app_id = create_deploy_oracle(public_key, private_key)
    print("Created application ID is: ", app_id)

    # Save the deployed application ID
    with open('oracle_contract.csv', 'w', newline='') as csvfile:
        my_writer = csv.writer(csvfile)
        my_writer.writerow(["APP_ID"])
        my_writer.writerow([app_id])




