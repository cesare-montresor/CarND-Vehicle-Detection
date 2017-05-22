import model as md
import matplotlib.pyplot as plt
import utils

predict_weights = "./models/cls_cnn_20170519-120820_03-0.01811-0.98367.h5"

model = md.classifierCNN((64,64,3),load_weights=predict_weights)


examples = [
    utils.loadImage('./datasources/non-vehicles/Extras/extra74.png'),
    utils.loadImage('./datasources/non-vehicles/Extras/extra66.png'),
    utils.loadImage('./datasources/non-vehicles/Extras/extra67.png'),
    utils.loadImage('./datasources/non-vehicles/Extras/extra112.png'),
    utils.loadImage('./datasources/non-vehicles/Extras/extra110.png')
]

preds = []

fig = plt.figure(figsize=(12, 3))

for i, ex in enumerate(examples):
    pred = model.predict(ex[None, :, :, :], batch_size=1)[0,0,0,0]
    plt.subplot(151+i)
    plt.imshow(ex)
    plt.title(str(pred))

plt.show()