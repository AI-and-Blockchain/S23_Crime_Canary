from pyteal import *

''' This is the oracle's smart contract. The ML web app sends a transaction to this smart contract to notify it that it has
a classification result to send to the token payment contract'''

def approval_program():
    on_creation = Seq(              #The Seq command executes each command in order
        [
            App.globalPut(Bytes("Owner"), Txn.sender()), #This creates a global state variable "Owner" to record the admin (the address that deploys the contract)
            Return(Int(1))   #Returning Int(1) is equivalent to approving the sequence
        ]
    )

    is_contract_admin = Txn.sender() == App.globalGet(Bytes("Owner"))

    classification_notify = Seq(
        [
           Assert(         #Checks that the subsequent conditions are met and terminates the sequence if they are not
               And(
                   Txn.note() == Bytes("classification-notification"),   #Transaction note used in calling the contract (useful for indexer operation)
                   Txn.application_args.length() == Int(3)  #Only three arguments must be passed during the contract call
                                                            #[transaction_call_type, prediction, severity]
               )
           ),
           #Update local state (prediction and severity) for the participant's address
           App.localPut(Int(1), Bytes("prediction"), Btoi(Txn.application_args[1])),  #Int(1) refers to the participant address
           App.localPut(Int(1), Bytes("severity"), Btoi(Txn.application_args[2])),

            Return(Int(1))   #Approve the sequence
        ]
    )

    program = Cond(
        [Txn.application_id() == Int(0), on_creation],  #Creates the application if appid is 0 during the contract call
        [Txn.on_completion() == OnComplete.DeleteApplication, Return(is_contract_admin)],  #Only admin can delete the application
        [Txn.on_completion() == OnComplete.UpdateApplication, Return(is_contract_admin)],  #Only admin can update the application
        [Txn.on_completion() == OnComplete.CloseOut, Return(Int(1))],  #Approve closing out the application
        [Txn.on_completion() == OnComplete.OptIn, Return(Int(1))], #Approve client opt-in to application
        [Txn.application_args[0] == Bytes("notify_classification"), classification_notify]  #Execute the classification_notify sequence if the first argument
                                                                                            #in the contract call is "notify_classification"
    )
    return program



def clear_state_program():
    # define the clear state program
    program = Seq(
        [
            Return(Int(1))  #Approve
        ]
    )
    return program

