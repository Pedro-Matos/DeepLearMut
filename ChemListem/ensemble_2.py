

from chemlistem import get_ensemble_model
model = get_ensemble_model()
results = model.process("The morphine was dissolved in ethyl acetate.", 0.0001, True)
for r in results: print(r)