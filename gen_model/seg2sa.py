from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image
import torch
from tqdm import tqdm
import os

def seg2sa(device, dataloader, output_folder, prompt):
    """ Generate satellite images from segmentation images using Seg2Sat model. 
    Args:
        device: The device to run the model on.
        dataloader: The dataloader containing the segmentation images.
        output_folder: The folder to save the generated satellite images.
        prompt: The prompt to generate the satellite images. 
    """


    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] == 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    controlnet = ControlNetModel.from_pretrained("rgres/Seg2Sat-sd-controlnet", torch_dtype=dtype)

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base", controlnet=controlnet, torch_dtype=dtype, safety_checker=None
    ).to(device)

    synt_images = []

    print("Processing images...")

    for batch_idx, (segsat, _, _, names, _, _) in enumerate(tqdm(dataloader)):
        # Only process 64 * 8 images
        if batch_idx >= 63:
            break
        
        for img_idx, name_tuple in enumerate(names):
            # Extract segmentation from segsat
            segmentation = segsat[img_idx].permute(1, 2, 0).cpu().numpy()
            segmentation = Image.fromarray((segmentation * 255).astype('uint8')).convert("RGB")

            # Generate image
            image = pipe(
                prompt=prompt,
                num_inference_steps=20,
                image=segmentation
            ).images[0]

            # Append to synt_images
            synt_images.append(image)

            # Create the filename
            name = str(name_tuple)
            filename = f"{name}.png"
            save_path = os.path.join(output_folder, filename)

            # Save the image
            image.save(save_path)

    return synt_images