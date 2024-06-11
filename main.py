from CNN import PersonFeatureExtractor
from Data import PersonDataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from Data import PersonDataset
from CNN import PersonFeatureExtractor


from torch.utils.data.dataloader import default_collate



root_dir = '.'
dataset = PersonDataset(root_dir)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

model = PersonFeatureExtractor(num_classes=5)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 15
for epoch in range(num_epochs):
    for images, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

torch.save(model, 'person_feature_extractor.pth')