import matplotlib.pyplot as plt


class History:

    def __init__(self):
        self.history = {
            'train_acc': [],
            'val_acc': [],
            'train_loss': [],
            'val_loss': [],
            'lr': [],
            'covid_recall_train': [],
            'covid_recall_valid': [],
            'covid_accuracy_train': [],
            'covid_accuracy_valid': [],
        }

    def save(self, train_acc, val_acc, train_loss, val_loss, lr, covid_recall_train, covid_recall_valid, covid_accuracy_train, covid_accuracy_valid):
        self.history['train_acc'].append(train_acc)
        self.history['val_acc'].append(val_acc)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['lr'].append(lr)
        self.history['covid_recall_train'].append(covid_recall_train)
        self.history['covid_recall_valid'].append(covid_recall_valid)
        self.history['covid_accuracy_train'].append(covid_accuracy_train)
        self.history['covid_accuracy_valid'].append(covid_accuracy_valid)

    def display_accuracy(self):
        epoch = len(self.history['train_acc'])
        epochs = [x for x in range(1, epoch + 1)]
        plt.title('Training accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.plot(epochs, self.history['train_acc'], label='Train')
        plt.plot(epochs, self.history['val_acc'], label='Validation')
        plt.legend()
        plt.show()

    def display_loss(self):
        epoch = len(self.history['train_acc'])
        epochs = [x for x in range(1, epoch + 1)]
        plt.title('Training loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.plot(epochs, self.history['train_loss'], label='Train')
        plt.plot(epochs, self.history['val_loss'], label='Validation')
        plt.legend()
        plt.show()

    def display_lr(self):
        epoch = len(self.history['train_acc'])
        epochs = [x for x in range(1, epoch + 1)]
        plt.title('Learning rate')
        plt.xlabel('Epochs')
        plt.ylabel('Lr')
        plt.plot(epochs, self.history['lr'], label='Lr')
        plt.show()

    def display(self):
        epoch = len(self.history['train_acc'])
        epochs = [x for x in range(1, epoch + 1)]

        fig, axes = plt.subplots(3, 1)
        plt.tight_layout()

        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Accuracy')
        axes[0].plot(epochs, self.history['train_acc'], label='Train')
        axes[0].plot(epochs, self.history['val_acc'], label='Validation')
        axes[0].legend()

        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Loss')
        axes[1].plot(epochs, self.history['train_loss'], label='Train')
        axes[1].plot(epochs, self.history['val_loss'], label='Validation')

        axes[2].set_xlabel('Epochs')
        axes[2].set_ylabel('Lr')
        axes[2].plot(epochs, self.history['lr'], label='Lr')

        plt.show()

        fig, axes = plt.subplots(2, 1)
        plt.tight_layout()

        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Covid Recall')
        axes[0].plot(epochs, self.history['covid_recall_train'], label='Train')
        axes[0].plot(epochs, self.history['covid_recall_valid'], label='Validation')
        axes[0].legend()

        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Covid Accuracy')
        axes[1].plot(epochs, self.history['covid_accuracy_train'], label='Train')
        axes[1].plot(epochs, self.history['covid_accuracy_valid'], label='Validation')

        plt.show()
