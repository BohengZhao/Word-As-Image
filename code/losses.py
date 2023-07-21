import torch.nn as nn
import torchvision
from scipy.spatial import Delaunay
import torch
import numpy as np
from torch.nn import functional as nnf
from easydict import EasyDict
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import torchvision.transforms as T
from diffusers import StableDiffusionPipeline
from IF_pipe import IFPipeline
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torchvision.transforms as tvt
import open_clip
from utils import *
from FontClassifier import FontSimilarityClassifier

from PIL import Image


class SDSLoss(nn.Module):
    def __init__(self, cfg, device):
        super(SDSLoss, self).__init__()
        self.cfg = cfg
        self.device = device
        self.pipe = StableDiffusionPipeline.from_pretrained(cfg.diffusion.model,
                                                            torch_dtype=torch.float16, use_auth_token=cfg.token)
        self.pipe = self.pipe.to(self.device)
        # default scheduler: PNDMScheduler(beta_start=0.00085, beta_end=0.012,
        # beta_schedule="scaled_linear", num_train_timesteps=1000)
        self.alphas = self.pipe.scheduler.alphas_cumprod.to(self.device)
        self.sigmas = (1 - self.pipe.scheduler.alphas_cumprod).to(self.device)

        self.text_embeddings = None
        self.embed_text()

    def embed_text(self):
        # tokenizer and embed text
        text_input = self.pipe.tokenizer(self.cfg.caption, padding="max_length",
                                         max_length=self.pipe.tokenizer.model_max_length,
                                         truncation=True, return_tensors="pt")
        uncond_input = self.pipe.tokenizer([""], padding="max_length",
                                           max_length=text_input.input_ids.shape[-1],
                                           return_tensors="pt")
        with torch.no_grad():
            text_embeddings = self.pipe.text_encoder(
                text_input.input_ids.to(self.device))[0]
            uncond_embeddings = self.pipe.text_encoder(
                uncond_input.input_ids.to(self.device))[0]
        self.text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        self.text_embeddings = self.text_embeddings.repeat_interleave(
            self.cfg.batch_size, 0)
        del self.pipe.tokenizer
        del self.pipe.text_encoder

    def forward(self, x_aug):
        sds_loss = 0

        # encode rendered image
        x = x_aug * 2. - 1.
        with torch.cuda.amp.autocast():
            init_latent_z = (self.pipe.vae.encode(x).latent_dist.sample())
        latent_z = 0.18215 * init_latent_z  # scaling_factor * init_latents

        with torch.inference_mode():
            # sample timesteps
            timestep = torch.randint(
                low=50,
                # avoid highest timestep | diffusion.timesteps=1000
                high=min(950, self.cfg.diffusion.timesteps) - 1,
                size=(latent_z.shape[0],),
                device=self.device, dtype=torch.long)

            # add noise
            eps = torch.randn_like(latent_z)
            # zt = alpha_t * latent_z + sigma_t * eps
            noised_latent_zt = self.pipe.scheduler.add_noise(
                latent_z, eps, timestep)

            # denoise
            # expand latents for classifier free guidance
            z_in = torch.cat([noised_latent_zt] * 2)
            timestep_in = torch.cat([timestep] * 2)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                print(z_in.shape)
                print(timestep.shape)
                print(self.text_embeddings.shape)
                eps_t_uncond, eps_t = self.pipe.unet(
                    z_in, timestep, encoder_hidden_states=self.text_embeddings).sample.float().chunk(2)

            eps_t = eps_t_uncond + self.cfg.diffusion.guidance_scale * \
                (eps_t - eps_t_uncond)

            # w = alphas[timestep]^0.5 * (1 - alphas[timestep]) = alphas[timestep]^0.5 * sigmas[timestep]
            w = self.alphas[timestep]**0.5 * self.sigmas[timestep]
            w = w[:, None, None, None]
            grad_z = w * (eps_t - eps)
            assert torch.isfinite(grad_z).all()
            grad_z = torch.nan_to_num(grad_z.detach().float(), 0.0, 0.0, 0.0)

        sds_loss = grad_z.clone() * latent_z
        del grad_z

        sds_loss = sds_loss.sum(1).mean()
        return sds_loss
    

class IFLoss(nn.Module):
    def __init__(self, cfg, device, flip=True):
        super(IFLoss, self).__init__()
        self.cfg = cfg
        if cfg.target_char.islower():
            self.prompt = "A black and white image of the lower case letter " + cfg.target_char
        else:
            self.prompt = "A black and white image of the upper case letter " + cfg.target_char
        self.device = device
        self.pipe = IFPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
        self.pipe.enable_model_cpu_offload()
        self.flip = flip
        self.transform = lambda x: torchvision.transforms.functional.rotate(x, 180)
        #self.pipe = self.pipe.to(self.device)
        # default scheduler: PNDMScheduler(beta_start=0.00085, beta_end=0.012,
        # beta_schedule="scaled_linear", num_train_timesteps=1000)
        self.alphas = self.pipe.scheduler.alphas_cumprod.to(self.device)
        self.sigmas = (1 - self.pipe.scheduler.alphas_cumprod).to(self.device)

    def forward(self, x_aug):
        sds_loss = 0
        batch_size = x_aug.shape[0]
        if self.flip:
            images = self.transform(x_aug)
        else:
            images = x_aug
        prompt_embeds, negative_embeds = self.pipe.encode_prompt(self.prompt, do_classifier_free_guidance=True)
        dtype = prompt_embeds.dtype
        prompt_embeds = torch.cat([negative_embeds, prompt_embeds])
        prompt_embeds = prompt_embeds.repeat_interleave(batch_size, 0)
        
        images = torch.nn.functional.interpolate(images, size=self.pipe.unet.sample_size, mode="bilinear", antialias=True)

        timestep = torch.randint(50, 950, (batch_size,), device=self.device)
        images.to(self.device)
        intermediate_images, eps = self.pipe.prepare_intermediate_images(
            images, timestep, batch_size, 1, dtype, self.device
        )

        z_in = torch.cat([intermediate_images] * 2) # ([10, 3, 64, 64])
        timestep_in = torch.cat([timestep] * 2)
        z_in = self.pipe.scheduler.scale_model_input(z_in, timestep_in)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            # check dimension of unet output
            noise_pred = self.pipe.unet(
                z_in, timestep, encoder_hidden_states=prompt_embeds)[0]
        guidance_scale = 100.0
        
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred_uncond, _ = noise_pred_uncond.split(z_in.shape[1], dim=1)
        noise_pred_text, predicted_variance = noise_pred_text.split(z_in.shape[1], dim=1)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        w = self.alphas[timestep]**0.5 * self.sigmas[timestep]
        w = w[:, None, None, None]
        grad_z = w * (noise_pred - eps)
        assert torch.isfinite(grad_z).all()
        grad_z = torch.nan_to_num(grad_z.detach().float(), 0.0, 0.0, 0.0)
        sds_loss = grad_z.clone().to(images.device) * images
        del grad_z
        sds_loss = sds_loss.sum(1).mean()
        return sds_loss


class ContrastiveLoss(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.temperature = 0.07
        # self.log_softmax = nn.LogSoftmax(dim=-1)
        self.img_loss = nn.CrossEntropyLoss()
        self.text_loss = nn.CrossEntropyLoss()

    def forward(self, text_embeddings, image_embeddings):
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        ground_truth = torch.arange(logits.shape[0]).to(logits.device)
        img_loss = self.img_loss(logits, ground_truth)
        text_loss = self.text_loss(logits.T, ground_truth)
        return (img_loss + text_loss) / 2


class ToneLoss(nn.Module):
    def __init__(self, cfg):
        super(ToneLoss, self).__init__()
        self.dist_loss_weight = cfg.loss.tone.dist_loss_weight
        self.im_init = None
        self.cfg = cfg
        self.mse_loss = nn.MSELoss()
        self.blurrer = torchvision.transforms.GaussianBlur(
            kernel_size=(cfg.loss.tone.pixel_dist_kernel_blur, cfg.loss.tone.pixel_dist_kernel_blur),
            sigma=(cfg.loss.tone.pixel_dist_sigma))

    def set_image_init(self, im_init):
        self.im_init = im_init.permute(2, 0, 1).unsqueeze(0)
        self.init_blurred = self.blurrer(self.im_init)

    def get_scheduler(self, step=None):
        if step is not None:
            return self.dist_loss_weight * np.exp(-(1/5)*((step-300)/(20)) ** 2)
        else:
            return self.dist_loss_weight

    def forward(self, cur_raster, step=None):
        blurred_cur = self.blurrer(cur_raster)
        return self.mse_loss(self.init_blurred.detach(), blurred_cur) * self.get_scheduler(step)


class ConformalLoss:
    def __init__(self, parameters: EasyDict, device: torch.device, target_letter: str, shape_groups):
        self.parameters = parameters
        self.target_letter = target_letter
        self.shape_groups = shape_groups
        self.faces = self.init_faces(device)
        self.faces_roll_a = [torch.roll(self.faces[i], 1, 1)
                             for i in range(len(self.faces))]
        self.device = device

        with torch.no_grad():
            self.angles = []
            self.reset()

    def get_angles(self, points: torch.Tensor) -> torch.Tensor:
        angles_ = []
        for i in range(len(self.faces)):
            triangles = points[self.faces[i]]
            triangles_roll_a = points[self.faces_roll_a[i]]
            edges = triangles_roll_a - triangles
            length = edges.norm(dim=-1)
            edges = edges / (length + 1e-1)[:, :, None]
            edges_roll = torch.roll(edges, 1, 1)
            cosine = torch.einsum('ned,ned->ne', edges, edges_roll)
            angles = torch.arccos(cosine)
            angles_.append(angles)
        return angles_

    def get_letter_inds(self, letter_to_insert):
        for group, l in zip(self.shape_groups, self.target_letter):
            if l == letter_to_insert:
                letter_inds = group.shape_ids
                return letter_inds[0], letter_inds[-1], len(letter_inds)

    def reset(self):
        points = torch.cat([point.clone().detach()
                           for point in self.parameters.point]).to(self.device)
        self.angles = self.get_angles(points)

    def init_faces(self, device: torch.device) -> torch.tensor:
        faces_ = []
        for j, c in enumerate(self.target_letter):
            points_np = [self.parameters.point[i].clone().detach().cpu().numpy()
                         for i in range(len(self.parameters.point))]
            start_ind, end_ind, shapes_per_letter = self.get_letter_inds(c)
            print(c, start_ind, end_ind)
            holes = []
            if shapes_per_letter > 1:
                holes = points_np[start_ind+1:end_ind]
            poly = Polygon(points_np[start_ind], holes=holes)
            poly = poly.buffer(0)
            points_np = np.concatenate(points_np)
            faces = Delaunay(points_np).simplices
            is_intersect = np.array(
                [poly.contains(Point(points_np[face].mean(0))) for face in faces], dtype=np.bool)
            faces_.append(torch.from_numpy(
                faces[is_intersect]).to(device, dtype=torch.int64))
        return faces_

    def __call__(self) -> torch.Tensor:
        loss_angles = 0
        points = torch.cat(self.parameters.point).to(self.device)
        angles = self.get_angles(points)
        for i in range(len(self.faces)):
            loss_angles += (nnf.mse_loss(angles[i], self.angles[i]))
        return loss_angles


class CLIPLoss(nn.Module):
    def __init__(self, device: torch.device, target_text: str, init_text: str, DUAL=True):
        super(CLIPLoss, self).__init__()
        self.render_width = 224
        self.render_height = 224
        # self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='laion400m_e32')
        # self.tokenizer = open_clip.get_tokenizer('ViT-L-14')
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-g-14', pretrained='laion2b_s12b_b42k')
        self.tokenizer = open_clip.get_tokenizer('ViT-g-14')

        self.model.requires_grad_(False)
        self.model.to(device)
        self.target_text = self.tokenizer(["a word of '" + target_text + "'"])
        if DUAL:
            self.init_text = self.tokenizer(["a word of '" + init_text + "'"])
        self.transform = lambda x: torchvision.transforms.functional.affine(img=x, angle=180.0, translate=(
            0, 0), scale=1.0, shear=0.0, interpolation=T.InterpolationMode.BILINEAR)
        # self.transform = lambda x: torch.rot90(x, 2, [-2, -1]) ## torch transform affine
        self.DUAL = DUAL
        self.device = device

    def forward(self, img):
        sum_loss = 0.0
        img = diff_norm(img, [0.48145466, 0.4578275, 0.40821073],
                        [0.26862954, 0.26130258, 0.27577711])
        if self.DUAL:
            clip_loss = 0.0
            model_out = self.model(img.to(self.device),
                                   self.init_text.to(self.device))
            logits_per_image = model_out[0]
            logits_per_text = model_out[1]
            logits = torch.mean(logits_per_image @ logits_per_text.T)
            clip_loss = clip_loss - logits
            sum_loss = sum_loss + clip_loss.item()
            clip_loss.backward(retain_graph=True)

        clip_loss = 0.0
        model_out = self.model(self.transform(img).to(
            self.device), self.target_text.to(self.device))
        logits_per_image = model_out[0]
        logits_per_text = model_out[1]
        logits = torch.mean(logits_per_image @ logits_per_text.T)
        clip_loss = clip_loss - logits
        sum_loss = sum_loss + clip_loss.item()
        return clip_loss


class TrOCRLoss(nn.Module):
    def __init__(self, device, target_text, batch_size):
        super(TrOCRLoss, self).__init__()
        self.processor = TrOCRProcessor.from_pretrained(
            "microsoft/trocr-base-handwritten")
        self.model = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-base-stage1")
        self.device = device
        self.transform = lambda x: torchvision.transforms.functional.affine(img=x, angle=180.0, translate=(
            0, 0), scale=1.0, shear=0.0, interpolation=T.InterpolationMode.BILINEAR)
        # set special tokens used for creating the decoder_input_ids from the labels
        self.model.config.decoder_start_token_id = self.processor.tokenizer.cls_token_id
        self.model.config.pad_token_id = self.processor.tokenizer.pad_token_id
        # make sure vocab size is set correctly
        self.model.config.vocab_size = self.model.config.decoder.vocab_size
        # set beam search parameters
        self.model.config.eos_token_id = self.processor.tokenizer.sep_token_id
        self.model.config.max_length = 64
        self.model.config.early_stopping = True
        self.model.config.no_repeat_ngram_size = 3
        self.model.config.length_penalty = 2.0
        self.model.config.num_beams = 4
        self.model.requires_grad_(False)
        self.model.to(device)
        self.labels = self.processor.tokenizer(
            target_text, padding="max_length", max_length=64).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        self.labels = torch.tensor(
            [label if label != self.processor.tokenizer.pad_token_id else -100 for label in self.labels]).unsqueeze(
            0)  # include batch size of 1
        self.labels = self.labels.expand(batch_size, -1)

    def forward(self, x):
        labels = self.labels.to(self.device)
        img = diff_norm(x, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        input_dict = {"pixel_values": self.transform(img), "labels": labels}
        outputs = self.model(**input_dict)
        loss = outputs.loss
        return loss


class FontClassLoss(nn.Module):
    def __init__(self, device, model_state, target_text, target_font, batch_size, rotate=True) -> None:
        super(FontClassLoss, self).__init__()
        self.device = device
        self.model = FontSimilarityClassifier()
        self.model.load_state_dict(model_state)
        self.model.eval()
        self.model.to(device)
        self.similarity_criterion = ContrastiveLoss()
        self.classification_criterion = nn.CrossEntropyLoss()
        self.target_text = target_text
        self.target_font = target_font
        #self.transform = lambda x: torchvision.transforms.functional.affine(img=x, angle=180.0, translate=(
        #    0, 0), scale=1.0, shear=0.0, interpolation=T.InterpolationMode.BILINEAR)
        self.transform = lambda x: torch.flip(x, dims=[0, 1])
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.font_name = self.tokenizer(target_font).expand(batch_size, -1)
        self.label = torch.tensor(self.char_mapping(target_text)).repeat(batch_size)
        self.rotate = rotate

    def char_mapping(self, char):
        if char.isupper():
            return ord(char) - ord('A')
        elif char.islower():
            return ord(char) - ord('a') + 26 # if the number of classes is 52
            #return ord(char) - ord('a') if number of class = 26

    def forward(self, x):
        # with torch.no_grad():
        if self.rotate:
            img = self.transform(x)
        else:
            img = x
        img = img.to(self.device)
        self.font_name = self.font_name.to(self.device)
        text_embeddings, image_embeddings, classification = self.model(img, self.font_name)
        similarity = text_embeddings@image_embeddings.T
        self.label = self.label.to(self.device)
        classification_loss = self.classification_criterion(classification, self.label)
        total_loss = classification_loss
        return total_loss


if __name__ == '__main__':
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load('/scratch/gilbreth/zhao969/FontClassifier/saved_checkpoints/checkpoint_500.pt')
    model_state = checkpoint['model_state_dict']
    loss = FontClassLoss(device, model_state, 'a', 'Arial')
    demo = Image.open("/scratch/gilbreth/zhao969/FontClassifier/data/output/Zephyrean_Gust_BRK_u.png")
    x = demo.convert('RGB')
    # x = torch.randn(1, 3, 224, 224)
    print(loss(x))
    '''
    from config import set_config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = set_config()
    sds_loss = SDSLoss(cfg, device)
    batch_size = 5
    x_aug = torch.randn(batch_size, 3, 224, 224, dtype=torch.float16).to(device)
    loss = sds_loss(x_aug)
    print(loss)

