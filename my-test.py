import ctsb
"""import jax.numpy as np """
import numpy as np 
import matplotlib.pyplot as plt



print(ctsb.model_registry.list_ids())

T = 100
p, q = 3, 3
problem = ctsb.problem("ARMA-v0")
cur_x = problem.initialize(p, q)
model = ctsb.model("ArmaOgd")
model.initialize(5)
model2 = ctsb.model("ArmaAdagrad")
model2.initialize(5)
loss = lambda y_true, y_pred: (y_true - y_pred)**2

results = []
results2 = []
for i in range(T):
    cur_y_pred = model.predict(cur_x)
    cur_y_pred2 = model2.predict(cur_x)
    cur_y_true = problem.step()
    cur_loss = loss(cur_y_true, cur_y_pred)
    cur_loss2 = loss(cur_y_true, cur_y_pred2)

    results.append(cur_loss)
    results2.append(cur_loss2)
    model.update(cur_loss)
    model2.update(cur_loss2)
    cur_x = cur_y_true


plt.plot(results,"r",results2,"b")
plt.title("AR-10 model on ARMA problem")
plt.show()





