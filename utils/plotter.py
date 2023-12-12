import matplotlib.pyplot as plt
import os
import numpy as np

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

    def plot_severity_vs_robustness(self, severities, natural_accuracies, robustness_accuracies, train_noise, plot_name='severity_vs_robustness.png'):
        """
        Plots severity vs average robustness accuracy.
        :param severities: List of severities.
        :param natural_accuracies: List of natural accuracy for each severity.
        :param robustness_accuracies: List of average robustness accuracy for each severity.
        :param train_noise: The type of noise the model was trained on.
        :param plot_name: Filename for the saved plot.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(severities, robustness_accuracies, marker='o', label=f'{train_noise} Train Noise')
        plt.title(f'Severity vs Robustness Accuracy for {train_noise} Noise')
        plt.xlabel('Severity')
        plt.ylabel('Robustness Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plot_dir, f'{train_noise}_{plot_name}'))
        plt.close()

    def plot_eval_noise_bar_chart(self, eval_noises, severity_accuracies, train_noise, plot_name='eval_noise_bar_chart.png'):
        """
        Bar plot where x axis is the eval noise and for each eval noise, there are bars for each severity.
        :param eval_noises: List of evaluation noises.
        :param severity_accuracies: Dictionary with keys as severities and values as lists of accuracies for each eval noise.
        :param train_noise: The type of noise the model was trained on.
        :param plot_name: Filename for the saved plot.
        """
        n_groups = len(eval_noises)
        fig, ax = plt.subplots(figsize=(15, 7))
        
        index = np.arange(n_groups)
        bar_width = 0.1
        opacity = 0.8
        
        for i, severity in enumerate(sorted(severity_accuracies.keys())):
            plt.bar(index + i * bar_width, severity_accuracies[severity], bar_width, alpha=opacity, label=f'Severity {severity}')

        plt.xlabel('Eval Noise')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy by Eval Noise and Severity for {train_noise} Noise')
        plt.xticks(index + bar_width, eval_noises)
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        plt.savefig(os.path.join(self.plot_dir, f'{train_noise}_{plot_name}'))
        plt.close()

# Example usage
if __name__ == "__main__":
    plotter = Plotter()
    # Dummy data for demonstration
    plotter.plot_loss_accuracy([0.9, 0.7, 0.5, 0.4], [0.6, 0.7, 0.8, 0.9])
