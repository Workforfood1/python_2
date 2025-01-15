import torch
import torch.nn as nn


def logistic_regression(train_dataLoader, input_size, num_classes, num_epochs, learning_rate,
                        loss, optimizer):
    class LogisticRegression(nn.Module):
        def __init__(self, input_size, num_classes):
            super(LogisticRegression, self).__init__()
            self.linear = nn.Linear(input_size, num_classes)

        def forward(self, feature):
            output = self.linear(feature)
            return output

    model = LogisticRegression(input_size, num_classes)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # train
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_dataLoader):
            images = torch.autograd.Variable(images.view(-1, input_size))
            labels = torch.autograd.Variable(labels)

            optimizer.zero_grad()
            output = model(images)
            compute_loss = loss(output, labels)
            compute_loss.backward()
            optimizer.step()
