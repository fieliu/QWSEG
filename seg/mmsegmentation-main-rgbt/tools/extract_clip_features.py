import argparse
import os
import torch


MFNET_CLASSES = ['unlabeled', 'car', 'person', 'bike', 'curve',
                 'car_stop', 'guardrail', 'color_cone', 'bump']

FMB_CLASSES = ['unlabeled', 'car', 'person', 'bicycle', 'curve',
               'car_stop', 'guardrail', 'color_cone', 'bump']

PST_CLASSES = ['unlabeled', 'fire', 'smoke', 'person', 'vehicle']

DATASET_CLASSES = {
    'mf': MFNET_CLASSES,
    'fmb': FMB_CLASSES,
    'pst': PST_CLASSES,
}


def extract_clip_features(dataset, clip_model_name='ViT-L/14', output_dir=None):
    import clip

    classes = DATASET_CLASSES[dataset]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, _ = clip.load(clip_model_name, device=device)
    model.eval()

    templates = [
        'a photo of a {}.', 'a photograph of a {}.',
        'an image of a {}.', 'a picture of a {}.',
        'the photo of a {}.', 'the photograph of a {}.',
        'the image of a {}.', 'the picture of a {}.',
    ]

    text_embeds_list = []
    for template in templates:
        texts = [template.format(name) for name in classes]
        tokens = clip.tokenize(texts).to(device)
        with torch.no_grad():
            features = model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        text_embeds_list.append(features)

    text_features = torch.stack(text_embeds_list).mean(dim=0)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    if output_dir is None:
        if dataset == 'mf':
            output_dir = '/home/lh/code/data/MFNet/text_embeddings'
        elif dataset == 'fmb':
            output_dir = '/home/lh/code/data/FMB_ALL/text_embeddings'
        elif dataset == 'pst':
            output_dir = '/home/lh/code/data/PST900/text_embeddings'

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{dataset}_class_embedding.pt')
    torch.save(text_features.cpu(), output_path)
    print(f'Saved {dataset} class embeddings to {output_path}')
    print(f'Shape: {text_features.shape}, dtype: {text_features.dtype}')
    return text_features


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mf', choices=['mf', 'fmb', 'pst'])
    parser.add_argument('--clip_model', type=str, default='ViT-L/14')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()
    extract_clip_features(args.dataset, args.clip_model, args.output_dir)
