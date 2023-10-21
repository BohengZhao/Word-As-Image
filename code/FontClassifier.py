import torch
import antialiased_cnns
import torch.nn as nn
import open_clip
import torch.nn.functional as F
import myconfig

import matplotlib.pyplot as plt


class ImageEncoder(torch.nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        # self.net_config = myconfig.read_yaml(
        #     '/scratch/gilbreth/zhao969/FontClassifier/config/config.yaml')[1]
        self.net_config = myconfig.read_yaml(
            '/home/zhao969/Documents/saved_checkpoints/config.yaml')[1]
        self.resnet = antialiased_cnns.resnet50(pretrained=True)
        self.resnet_end = nn.Sequential(*list(self.resnet.children())[:-1])
        if self.net_config['Number_of_Classes'] == 52:
            self.class_head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(in_channels=2048, out_channels=26*2, kernel_size=1)
            )
        else:
            self.class_head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(in_channels=2048, out_channels=26, kernel_size=1)
            )

        self.img_embedding = nn.Linear(2048, 512)

    def forward(self, x):
        x = self.resnet_end(x)
        classification = self.class_head(x).squeeze(3).squeeze(2)
        img_embedding = self.img_embedding(x.squeeze(3).squeeze(2))
        return img_embedding, classification


'''
class ImageEncoder(torch.nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.resnet = antialiased_cnns.resnet18(pretrained=True)
        self.resnet_front = nn.Sequential(*list(self.resnet.children())[:5])
        self.resnet_end = nn.Sequential(*list(self.resnet.children())[5:-1])

        self.class_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels=64, out_channels=26*2, kernel_size=1)
        )

    def forward(self, x):
        inter_result = self.resnet_front(x)
        classification = self.class_head(inter_result).squeeze(3).squeeze(2)  # Squeeze the 1*1 HW dimension

        x = self.resnet_end(inter_result)
        img_embedding = torch.squeeze(x, 3).squeeze(2)
        return img_embedding, classification
'''


class FontnameEncoder(torch.nn.Module):
    def __init__(self):
        super(FontnameEncoder, self).__init__()
        self.net_config = myconfig.read_yaml(
            '/scratch/gilbreth/zhao969/FontClassifier/config/config.yaml')[1]
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='laion2b_s34b_b79k')  # text embdedding size = 512
        if self.net_config['train_CLIP'] == False:
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            for param in self.model.parameters():
                param.requires_grad = True
        # for param in self.model.parameters():
        #    param.requires_grad = False
        # self.tokenizer = open_clip.get_tokenizer(
        # 'ViT-B-32')  # Done in dataset loader

    def forward(self, text):
        text = text.squeeze(1)  # text.shape = [batch_size, 77]
        text_features = self.model.encode_text(text)
        return text_features


class FontSimilarityClassifier(nn.Module):
    def __init__(self):
        super(FontSimilarityClassifier, self).__init__()
        self.img_encoder = ImageEncoder()
        self.font_encoder = FontnameEncoder()

    def get_img_embedding(self, img):
        img_embed, _ = self.img_encoder(img)
        img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)
        return img_embed

    def get_text_embedding(self, text):
        text_embed = self.font_encoder(text)
        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)
        return text_embed

    def forward(self, img, text):
        img_embed, classification = self.img_encoder(img)
        text_embed = self.font_encoder(text)
        # img_embed = F.normalize(img_embed, p=2, dim=1)
        # text_embed = F.normalize(text_embed, p=2, dim=1)
        img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)
        text_embed = text_embed / text_embed.norm(dim=-1, keepdim=True)

        # Fix temperature constant (ref CLIP paper appendix)
        # logits = torch.matmul(img_embed, text_embed.T) * torch.exp(torch.tensor(0.07))
        return text_embed, img_embed, classification


if __name__ == '__main__':
    import torchvision.transforms as tvt
    from PIL import Image
    import kornia.augmentation as K

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FontSimilarityClassifier()
    model.to(device)
    checkpoint = torch.load('/home/zhao969/Documents/saved_checkpoints/checkpoint_500.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    transform = tvt.Compose([tvt.ToTensor()])
    # transform = tvt.Compose([tvt.ToTensor(), tvt.RandomAffine(degrees=10, translate=(0.2, 0.2), scale=(
    #   0.9, 1.1), shear=5, fill=0.0, interpolation=tvt.InterpolationMode.BILINEAR)])

    # transform = tvt.Compose([tvt.ToTensor(), tvt.RandomRotation(50, interpolation=tvt.InterpolationMode.BILINEAR, fill=0.0)])
    image = Image.open(
        '/scratch/gilbreth/zhao969/Word-As-Image/output/font_loss_upgrade_e/IndieFlower-Regular/IndieFlower-Regular_e_scaled_to_e_seed_0/output-png/output.png').convert('RGB')
    # image = tvt.functional.invert(image)
    # image = tvt.functional.invert(transform(image)).unsqueeze(0).to(device)
    '''
    augmentations = []
    augmentations.append(K.RandomPerspective(distortion_scale=0.5, p=0.7))
    augmentations.append(K.RandomCrop(size=(224, 224), pad_if_needed=True, p=1.0, fill=0.0))
    augment = nn.Sequential(*augmentations)
    image = augment.forward(transform(tvt.functional.invert(image))).to(device)
    print(image.shape)
    image = tvt.functional.invert(image)
    save_img = image[0].permute(1, 2, 0).detach().cpu().numpy()
    # plt.imshow(save_img)
    plt.imsave('Before_NN.png', save_img)
    print("Done")
    '''
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    fontname_token = tokenizer('2Toon Expanded')
    fontname = fontname_token.unsqueeze(0).to(device)
    image = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        text_embeddings, image_embeddings, classification = model(image, fontname)
    # print(classification)
    # print(torch.argmax(classification, dim=1))
    classification = classification[0].detach().cpu().numpy()
    plt.scatter([i for i in range(len(classification))], classification)
    for x, y in zip([i for i in range(len(classification))], classification):
        label = chr(ord('a') + x)
        plt.annotate(label,  # this is the text
                     (x, y),  # these are the coordinates to position the label
                     textcoords="offset points",  # how to position the text
                     xytext=(0, 10),  # distance from text to points (x,y)
                     ha='center')  # horizontal alignment can be left, right or center
    plt.savefig("Loss Plot")
