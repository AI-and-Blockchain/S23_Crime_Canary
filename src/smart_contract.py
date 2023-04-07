from pyteal import *

def approval_program():
    is_positive_bool = Txn.application_args[0] == Bytes("1")
    is_user_address = Txn.application_args[1] == Bytes("user-address")
    on_call = Seq([
        Assert(is_positive_bool),
        Assert(is_user_address),
        App.localPut(Bytes("token-asset"), App.id(0), Int(1)),  
        App.transfer(Int(1), Bytes("user-address"), Int(1), "token-asset"),
        Return()
    ])
    program = Cond(
        [Txn.application_id() == Int(0), Seq([App.localPut(Int(0), Bytes("admin-key"), Txn.sender()), Return()])],
        [Txn.on_completion() == OnComplete.NoOp, Return()],
        [Txn.on_completion() == OnComplete.OptIn, Return()],
        [Txn.on_completion() == OnComplete.CloseOut, Return()],
        [Txn.on_completion() == OnComplete.UpdateApplication, Return()],
        [Txn.on_completion() == OnComplete.DeleteApplication, Return()],
        [Txn.on_completion() == OnComplete.ClearState, Return()],
        [Txn.application_args.length() == Int(2), on_call]
    )
    return program

def clear_state_program():
    # define the clear state program
    return Seq([
        App.localDel(Bytes("token-asset"), App.id(0)),
        Return()
    ])
