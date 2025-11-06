#AER850 Project 2
#Bryant Berrio 501162030
#Model Test File

#Step 5 Model Testing
# p2_test.py — Step 5 (memory-safe single-load flow)

import os, json, glob, gc
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import load_model

# ----- paths
basedir = r"C:\Users\bryan\Documents\GitHub\AER850-BB-Project-2\Project 2 Data"
datadir = os.path.join(basedir, "Data")
valdir  = os.path.join(datadir, "valid")
testdir = os.path.join(datadir, "test")

# ----- cfg
imgsize   = (500,500)
batchsize = 32

# ----- class map
with open("class_indices.json","r") as jf:
    cls2idx = json.load(jf)
idx2cls = {v:k for k,v in cls2idx.items()}
class_list = [idx2cls[i] for i in range(len(idx2cls))]

# ----- val generator (for model pick)
val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    directory=valdir, target_size=imgsize, batch_size=batchsize,
    class_mode="categorical", shuffle=False
)
vsteps = max(1, val_gen.n // val_gen.batch_size)

# ----- helpers
def _clear():
    tf.keras.backend.clear_session()
    gc.collect()

def eval_model_once(path):
    """load -> evaluate -> free memory; returns (val_acc, val_loss)"""
    m = load_model(path, compile=True)     # try compile=False if still tight on RAM
    loss, acc = m.evaluate(val_gen, steps=vsteps, verbose=0)
    del m
    _clear()
    return acc, loss

# ----- pick best model without holding both in memory
m_a_path, m_b_path = "best_model_a.h5", "best_model_b.h5"
if not (os.path.exists(m_a_path) and os.path.exists(m_b_path)):
    raise FileNotFoundError("Missing best_model_a.h5 or best_model_b.h5 in working dir.")

acc_a, loss_a = eval_model_once(m_a_path)
acc_b, loss_b = eval_model_once(m_b_path)
print(f"val acc — A: {acc_a:.4f} | B: {acc_b:.4f}")

best_path = m_b_path if acc_b >= acc_a else m_a_path
best_name = "Model B" if acc_b >= acc_a else "Model A"
print(f"using {best_name}\n")

# ----- load only the winner for prediction
best_model = load_model(best_path, compile=False)  # compile not needed for predict()

# ----- choose one test image per class
prefer = {
    "crack":        ["test_crack.jpg","crack.jpg"],
    "missing-head": ["test_missinghead.jpg","missinghead.jpg","missing-head.jpg"],
    "paint-off":    ["test_paintoff.jpg","paintoff.jpg","paint-off.jpg"],
}
def pick_img_for(clsname):
    for nm in prefer.get(clsname, []):
        for root in (basedir, datadir, testdir, os.getcwd()):
            p = os.path.join(root, nm)
            if os.path.exists(p): return p
    patt = os.path.join(testdir, clsname, "*.*")
    files = sorted(glob.glob(patt))
    if not files:
        raise FileNotFoundError(f"no test images for class '{clsname}' in {patt}")
    return files[0]

test_pick = {c: pick_img_for(c) for c in class_list}

# ----- preprocess + predict
def prep(path):
    im = load_img(path, target_size=imgsize)
    arr = img_to_array(im)/255.0
    return np.expand_dims(arr,0), im

def predict_one(path):
    batch, pil = prep(path)
    probs = best_model.predict(batch, verbose=0)[0]
    k = int(np.argmax(probs))
    return pil, idx2cls[k], float(probs[k]), probs

# ----- run predictions
results = {}
for clsname in class_list:
    pth = test_pick[clsname]
    pil, pcls, pconf, _ = predict_one(pth)
    results[clsname] = dict(img=pil, pred=pcls, conf=pconf, path=pth)
    print(f"{os.path.basename(pth)} -> {pcls}  ({pconf:.1%})")

# ----- plot panel
plt.figure(figsize=(12,4))
for i, clsname in enumerate(class_list, start=1):
    r = results[clsname]
    plt.subplot(1,3,i)
    plt.imshow(r["img"]); plt.axis("off")
    ttl = f"True: {clsname}\nPred: {r['pred']} ({r['conf']:.1%})"
    plt.title(ttl, fontsize=10)
plt.tight_layout()
plt.savefig("step5_predictions.png", dpi=160)
plt.show()
print("saved: step5_predictions.png")

# free
del best_model
_clear()
