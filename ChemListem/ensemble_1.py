# first proposed problem: ChemListem uses three models:
#  - a "traditional" model
#  - a "minimalist" model
#  - ensemble model that combines the two. The following example shows how to use the ensemble model:

from chemlistem import get_ensemble_model
model = get_ensemble_model()
results = model.process("The morphine was dissolved in ethyl acetate.")
print(results)