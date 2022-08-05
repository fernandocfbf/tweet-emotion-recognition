from cProfile import label
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torch import normal

def show_history(h):
    epochs_trained = len(h.history["loss"])

    plt.figure(figsize=(16, 6))

    plt.subplot(1,2,1)
    plt.plot(range(0, epochs_trained), h.history.get("accuracy"), label="Training")
    plt.plot(range(0, epochs_trained), h.history.get("val_accuracy"), label="Validation")
    plt.ylim([0., 1.])
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(range(0, epochs_trained), h.history.get("loss"), label="Training")
    plt.plot(range(0, epochs_trained), h.history.get("val_loss"), label="Validation")
    plt.ylim([0., 1.])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def show_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, normalize=True)

    plt.figure(figsize=(8,8))
    sp = plt.subplot(1,1,1)
    ctx = sp.mathshow(cm)
    plt.xticks(list(range(0, 6)), labels=classes)
    plt.yticks(list(range(0, 6)), labels=classes)
    plt.colorbar(ctx)
    plt.show()

