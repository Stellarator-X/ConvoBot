import sys

def progress_bar(header, curr_iteration, total_iterations, num_bars=30, output_vals={}):
    """
        Shows a progress bar when called inside a loop
        params:
            curr_iteration, total_iterations : self_exp
            num_bars
            output_vals: dictionary of values to be displayed
    """
    num_bars = min(num_bars, total_iterations)
    total_iter = total_iterations
    total_iterations -= total_iterations%num_bars
    percent = min(100*curr_iteration/(total_iterations-1), 100.00)
    bars = round((num_bars*percent)/100)
    done = '[' + bars*'='
    todo = (num_bars - bars)*'~' + ']'
    valstring = ""
    for i, val in enumerate(output_vals):
        valstring += val + " : " + f"{(output_vals[val]):.2f}" + " "
    print(f"\r{header}:{done+todo}({percent:0.2f}%)  {valstring}", end = "")
    sys.stdout.flush()

    if(curr_iteration == total_iter-1):
        sys.stdout.flush()
        print()
    

if __name__ == "__main__":
    import time
    cost  = 3000
    acc = 0.9
    for i in range(1000):
        progress_bar("Progress",i, 1000, num_bars = 40, output_vals={"Cost":cost, "Accuracy":acc})
        cost/=3
        acc *= 1.001
        time.sleep(0.01)
