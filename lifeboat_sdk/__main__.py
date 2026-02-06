from lifeboat_sdk import Lifeboat

model = Lifeboat()
model.load()

result = model.predict(pclass=3, sex="male", age=25, sibsp=0, parch=0, fare=7.25, embarked="S")
print(f"Survival: {result['survival']}, Probability: {result['probability']:.4f}")
