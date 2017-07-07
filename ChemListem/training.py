from chemlistem import tradmodel
tm = tradmodel.TradModel()
tm.train("train_final.txt", "CEMP_mut.tsv", None, "tradtest")