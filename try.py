import torch
from preprocess import preprocess_image
from CNN import PersonFeatureExtractor

who = ['fanruoruo', 'liyunrui', 'silili', 'yelinger', 'yuanmeng']
model=torch.load('person_feature_extractor.pth')
model.eval()

image = preprocess_image("./target/2.jpg")
image_tensor = torch.from_numpy(image).float()
pred = model(image_tensor).argmax(dim=1)
print(who[pred])
