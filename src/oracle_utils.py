
import base64
from algosdk import transaction
from algosdk.v2client import algod
from pyteal import compileTeal, Mode
from oracle_smartcontract import approval_program, clear_state_program

''' This is the utility file for oracle operations'''

#function to compile program source
def compile_program(client, source_code):
    compile_response = client.compile(source_code)
    return base64.b64decode(compile_response["result"])


# helper function that waits for a given txid to be confirmed by the network
def wait_for_confirmation(client, txid):
    last_round = client.status().get("last-round")
    txinfo = client.pending_transaction_info(txid)
    while not (txinfo.get("confirmed-round") and txinfo.get("confirmed-round") > 0):
        print("Waiting for confirmation...")
        last_round += 1
        client.status_after_block(last_round)
        txinfo = client.pending_transaction_info(txid)
    print(
        "Transaction {} confirmed in round {}.".format(
            txid, txinfo.get("confirmed-round")
        )
    )
    return txinfo


# create new application
def create_app(client,public_key,private_key,approval_program,clear_program,global_schema,local_schema,app_args):
    # define sender as creator
    sender = public_key
    # declare on_complete as NoOp
    on_complete = transaction.OnComplete.NoOpOC.real
    # get node suggested parameters
    params = client.suggested_params()
    # create unsigned transaction
    txn = transaction.ApplicationCreateTxn(sender,params,on_complete,approval_program,clear_program,global_schema,
                                           local_schema,app_args,)

    # sign transaction
    signed_txn = txn.sign(private_key)
    tx_id = signed_txn.transaction.get_txid()
    # send transaction
    client.send_transactions([signed_txn])
    # await confirmation
    wait_for_confirmation(client, tx_id)

    # display results
    transaction_response = client.pending_transaction_info(tx_id)
    app_id = transaction_response["application-index"]
    print("Created new app-id:", app_id)

    return app_id


# opt-in to application
def opt_in_app(client, public_key, private_key, app_id):
    # declare sender
    sender = public_key
    print("OptIn from account: ", sender)

    # get node suggested parameters
    params = client.suggested_params()

    # create unsigned transaction
    txn = transaction.ApplicationOptInTxn(sender, params, app_id)
    # sign transaction
    signed_txn = txn.sign(private_key)
    tx_id = signed_txn.transaction.get_txid()
    # send transaction
    client.send_transactions([signed_txn])
    # await confirmation
    wait_for_confirmation(client, tx_id)

    # display results
    transaction_response = client.pending_transaction_info(tx_id)
    print("OptIn to app-id:", transaction_response["txn"]["txn"]["apid"])


# call application
def call_app(client, public_key, private_key, app_id, app_args, participant_address):
    # declare sender
    sender = public_key
    print("Call from account:", sender)

    # get node suggested parameters
    params = client.suggested_params()

    # create unsigned transaction
    txn = transaction.ApplicationCallTxn(sender=sender, sp=params, index=app_id, on_complete=transaction.OnComplete.NoOpOC,
                                         app_args=app_args, accounts=participant_address)

    # sign transaction
    signed_txn = txn.sign(private_key)
    tx_id = signed_txn.transaction.get_txid()
    # send transaction
    client.send_transactions([signed_txn])
    # await confirmation
    wait_for_confirmation(client, tx_id)


#Ignore this function for now
def format_state(state):
    formatted = {}
    for item in state:
        key = item["key"]
        value = item["value"]
        formatted_key = base64.b64decode(key).decode("utf-8")
        if value["type"] == 1:
            # byte string
            if formatted_key == "voted":
                formatted_value = base64.b64decode(value["bytes"]).decode("utf-8")
            else:
                formatted_value = value["bytes"]
            formatted[formatted_key] = formatted_value
        else:
            # integer
            formatted[formatted_key] = value["uint"]
    return formatted

#Ignore this function for now
# read user local state
def read_local_state(client, addr, app_id):
    results = client.account_info(addr)
    for local_state in results["apps-local-state"]:
        if local_state["id"] == app_id:
            if "key-value" not in local_state:
                return {}
            return format_state(local_state["key-value"])
    return {}

#Ignore this function for now
# read app global state
def read_global_state(client, addr, app_id):
    results = client.account_info(addr)
    apps_created = results["created-apps"]
    for app in apps_created:
        if app["id"] == app_id:
            return format_state(app["params"]["global-state"])
    return {}


# convert 64 bit integer i to byte string
def intToBytes(i):
    return i.to_bytes(8, "big")

''' This is the oracle's smart contract. The ML web app sends a transaction to this smart contract to notify it that it has
a classification result to send to the token payment contract'''



''' Function to create and deploy the Oracle smart contract to the Blockchain'''
def create_deploy_oracle(creator_public_key, creator_private_key):
    # initialize an algodClient
    #algod_token = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    #algod_address = "http://localhost:4001"
    #algod_client = algod.AlgodClient(algod_token, algod_address)
    algod_address = "https://testnet-algorand.api.purestake.io/ps2"
    algod_token = "3UeCpd3CVz9lvWElfm0sr3zQzvP6FyOC136j7WvQ"
    headers = {"X-API-Key": algod_token, }
    algod_client = algod.AlgodClient(algod_token, algod_address, headers)


    # declare application state storage (immutable)
    local_ints = 5
    local_bytes = 5
    global_ints = (5)  # 4 for setup + 20 for choices. Use a larger number for more choices.
    global_bytes = 5
    global_schema = transaction.StateSchema(global_ints, global_bytes)
    local_schema = transaction.StateSchema(local_ints, local_bytes)

    # get PyTeal approval program
    approval_program_ast = approval_program()
    # compile program to TEAL assembly
    approval_program_teal = compileTeal(approval_program_ast, mode=Mode.Application, version=2)
    # compile program to binary
    approval_program_compiled = compile_program(algod_client, approval_program_teal)

    # get PyTeal clear state program
    clear_state_program_ast = clear_state_program()
    # compile program to TEAL assembly
    clear_state_program_teal = compileTeal(clear_state_program_ast, mode=Mode.Application, version=2)
    # compile program to binary
    clear_state_program_compiled = compile_program(algod_client, clear_state_program_teal)

    app_args = []
    # create new application
    app_id = create_app(algod_client,creator_public_key,creator_private_key,approval_program_compiled,clear_state_program_compiled,global_schema,
                        local_schema,app_args)

    return app_id







