from chemlistem import tradmodel
tm = tradmodel.TradModel()
tm.load("tradmodel_tradtest.json", "tradmodel_tradtest.h5")
print(tm.process("The morphine was dissolved in ethyl acetate."))