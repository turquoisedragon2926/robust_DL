import matplotlib.pyplot as plt
import os

class Plotter:
    def __init__(self, plot_dir='plots'):
        """
        Initializes the plotter.
        :param plot_dir: Directory where plots will be saved.
        """
        self.plot_dir = plot_dir
        os.makedirs(plot_dir, exist_ok=True)

    def plot_loss_accuracy(self, loss, accuracy, cifar10c_eval_acc=None, plot_name='plot.png'):
        """
        Plots the training loss and validation accuracy.
        :param loss: List of loss values.
        :param accuracy: List of accuracy values.
        :param epoch: Current epoch number.
        :param plot_name: Filename for the saved plot.
        """
        plt.figure(figsize=(12, 5))
        plt.suptitle(plot_name)

        # Plot training loss
        plt.subplot(1, 2, 1)
        plt.plot(loss, label='Training Loss')
        plt.title('Loss vs. Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Plot validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(accuracy, label='Validation Accuracy')
        plt.title('Accuracy vs. Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        if cifar10c_eval_acc is not None:
            plt.axhline(y=cifar10c_eval_acc, color='r', linestyle='-', label='CIFAR 10 C EVAL')

        # Save the plot
        plt.savefig(os.path.join(self.plot_dir, plot_name))
        plt.close()

# Example usage
if __name__ == "__main__":
    plotter = Plotter()
    # Dummy data for demonstration
    plotter.plot_loss_accuracy([0.9, 0.7, 0.5, 0.4], [0.6, 0.7, 0.8, 0.9])
