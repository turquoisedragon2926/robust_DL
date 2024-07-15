def identity_attack(model, X, y):
  out = model(X)
  acc = (out.data.max(1)[1] == y.data).float().sum()
  return acc.item()
