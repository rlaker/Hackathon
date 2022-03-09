# unsorted imports
import pandas as pd
import numpy as np
import tqdm 
import matplotlib.pyplot as plt

awful_list = ["really long string so that it runs onto the next line", 1,2,3,4,5,6,7,8,9,10, "not there yet", "maybe now"]


def fun_with_too_many_args(arg1, arg2, arg3, arg4, arg5, arg6, arg7 = 2, arg8 = False, arg9 = True, another_one_to_go_past_limit= False):
    return arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, another_one_to_go_past_limit


def fun_too_close_to_another(list):
    # using pandas and numpy but not matplotlib,
    # it will remove unused imports

    pd.DataFrame()
    np.ones(1)
    return list
fun_too_close_to_another(awful_list)
