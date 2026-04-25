import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from models.cnn import SimpleCNN


def evaluate(model, testloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=64,
        shuffle=True
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=64,
        shuffle=False
    )

    model = SimpleCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5
    train_losses = []
    test_accuracies = []

    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        average_loss = running_loss / len(trainloader)
        accuracy = evaluate(model, testloader, device)

        train_losses.append(average_loss)
        test_accuracies.append(accuracy)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, "
            f"Loss: {average_loss:.4f}, "
            f"Test Accuracy: {accuracy:.2f}%"
        )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "saved_models/best_model.pth")

    print(f"Best Test Accuracy: {best_accuracy:.2f}%")

    plt.figure()
    plt.plot(train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Curve")
    plt.savefig("results/loss_curve.png")

    plt.figure()
    plt.plot(test_accuracies)
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Test Accuracy Curve")
    plt.savefig("results/accuracy_curve.png")


if __name__ == "__main__":
    main()