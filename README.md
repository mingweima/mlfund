# mlfund
Components:
1. pred: fundamental prediction algorithms. input: feature sets. output: EPS etc. 
2. val: valuation models. input: from pred. output: PV/Price for companies (score)
3. cnstr: portfolio construction. input: a score to rank stocks. output: permnos to L/S for industry
4. btest: backtesting tool. input: permnos (with or w/o industry) updated at some frequency. output: daily returns 
5. ret_analysis: returns analysis. input: daily returns df. output: analysis by or not by industry 
