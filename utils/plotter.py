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

        # Adjust layout to prevent overlap and provide space for suptitle
        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Set suptitle and adjust its position
        plt.suptitle(plot_name, y=0.98)
        
        # Save the plot
        plt.savefig(os.path.join(self.plot_dir, plot_name))
        plt.close()

    def plot_severity_vs_robustness(self, severities, natural_accuracies, robustness_accuracies, train_noise, metric="Severity", plot_name='severity_vs_robustness.png'):
        """
        Plots severity vs natural and average robustness accuracy.
        :param severities: List of severities.
        :param natural_accuracies: List of natural accuracy for each severity.
        :param robustness_accuracies: List of average robustness accuracy for each severity.
        :param train_noise: The type of noise the model was trained on.
        :param plot_name: Filename for the saved plot.
        """
        plt.figure(figsize=(10, 5))
        # Plot natural accuracies
        plt.plot(severities, natural_accuracies, marker='s', linestyle='--', color='green', label='Natural Accuracy')
        # Plot robustness accuracies
        plt.plot(severities, robustness_accuracies, marker='o', linestyle='-', color='blue', label=f'Average Robustness Accuracy')
        plt.title(f'{metric} vs Accuracy for {train_noise} Noise')
        plt.xlabel(f'{metric}')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plot_dir, f'{train_noise}_{plot_name}'))
        plt.close()

    def plot_combined_severity_vs_robustness(self, severities, robustness_accuracies, train_noises, metric="Severity", plot_name='combined_severity_vs_robustness.png', robust=True):
        """
        Plots severity vs natural and average robustness accuracy for all train_noises.
        :param severities: List of severities.
        :param natural_accuracies: List of natural accuracy for each severity.
        :param robustness_accuracies: Dictionary with keys as train_noise the model was trained on and values as lists of accuracies for each severity.
        :param train_noises: The type of noises the models were trained on.
        :param plot_name: Filename for the saved plot.
        """

        plt.figure(figsize=(10, 5))

        # Define a list of colors for the plot lines
        colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'brown']
        color_index = 0

        for noise in train_noises:
            # Check if the noise type is in the robustness_accuracies dictionary and if we have enough colors
            if noise in robustness_accuracies and color_index < len(colors):
                # Plot the robustness accuracies for each noise type with a specific color
                plt.plot(severities, robustness_accuracies[noise], marker='o', linestyle='-', label=f'{noise} Noise', color=colors[color_index])
                color_index += 1

        plt.title(f'{metric} vs Average {"Robustness" if robust else "Natural"} Accuracy for Different Noises')
        plt.xlabel(f'{metric}')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.plot_dir, plot_name))
        plt.close()

    def plot_eval_noise_bar_chart(self, eval_noises, severity_accuracies, train_noise, plot_name='eval_noise_bar_chart.png', metric="Severity"):
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
            plt.bar(index + i * bar_width, severity_accuracies[severity], bar_width, alpha=opacity, label=f'{metric} {severity}')

        plt.xlabel('Eval Noise')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy by Eval Noise and {metric} for {train_noise} Noise')

        # Rotate the x-axis labels to avoid overlapping
        plt.xticks(index + bar_width / 2, eval_noises, rotation=45, ha='center')

        plt.legend()

        # Adjust the subplot parameters to give the x-axis labels more space
        plt.subplots_adjust(bottom=0.15)

        plt.grid(True)
        plt.savefig(os.path.join(self.plot_dir, f'{train_noise}_{plot_name}'))
        plt.close()

    def plot_tradeoff(self, severities, natural_accuracies, robustness_accuracies, plot_name='tradeoff.png'):
        """
        Plots a Pareto frontier where we measure tradeoff in natural accuracy vs average robustness accuracy, 
        and each point represents a model, labeled with train_noise and the corresponding severity.
        :param severities: List of severities.
        :param natural_accuracies: Dictionary with keys as train_noise the model was trained on and natural accuracy for each severity as a list.
        :param robustness_accuracies: Dictionary with keys as train_noise the model was trained on and average robustness accuracy for each severity as a list.
        :param plot_name: Filename for the saved plot.
        """

        plt.figure(figsize=(10, 7))

        for train_noise, severities_natural_accuracies in natural_accuracies.items():
            for index, natural_accuracy in enumerate(severities_natural_accuracies):
                if train_noise in robustness_accuracies and len(robustness_accuracies[train_noise]) > index:
                    severity = severities[index]
                    robust_accuracy = robustness_accuracies[train_noise][index]
                    plt.scatter(natural_accuracy, robust_accuracy)
                    plt.annotate(f'{train_noise}, {severity}', (natural_accuracy, robust_accuracy))

        plt.title('Tradeoff in Natural Accuracy vs Average Robustness Accuracy')
        plt.xlabel('Natural Accuracy')
        plt.ylabel('Robustness Accuracy')
        plt.grid(True)
        plt.savefig(os.path.join(self.plot_dir, plot_name))
        plt.close()

# Example usage
if __name__ == "__main__":
    plotter = Plotter()
    # Dummy data for demonstration
    plotter.plot_loss_accuracy([0.9, 0.7, 0.5, 0.4], [0.6, 0.7, 0.8, 0.9])
