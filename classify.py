import numpy as np
import torch
import torchvision.transforms as transforms
from preprocessor.base_preprocessor import ColonPreprocessor
from postprocessor.knn_postprocessor import KNNPostprocessor

THRESHOLD_AD, THRESHOLD_HP = 0.898989, 0.888888
THRESHOLD_BEST = -0.7084716104996369 # -0.45629746306226204
THRESHOLD_ID, THRESHOLD_OOD = THRESHOLD_BEST+1, THRESHOLD_BEST-1
IMG_SIZE = 224

Label = {0: 'AD', 1: 'HP', 2: 'OOD'}
Confidence = {0: 'High Confidence', 1: 'Low Confidence'}

def classification(net_class, net_ood, image, device):
    conf = 1
    transform = ColonPreprocessor()
    postprocessor = KNNPostprocessor()
    net_class = net_class.cuda()
    net_class.eval()

    # Image Processing
    image = transform(image).unsqueeze(0).to(device) # to add batch
    # cropped_frame_tensor = transform_to_tensor(cropped_frame_resized).unsqueeze(0).to(device)

    # Step 1: Classification
    with torch.no_grad():
        logits = net_class(image)
        score = torch.softmax(logits, dim=1)
        confidence, classification = torch.max(score, dim=1)
    # Evaluation on the confidence
    print(f'AD Threshold: {THRESHOLD_AD}, HP Threshold: {THRESHOLD_HP}')
    print(f'Classification model: {classification}, {confidence}')
    
    # For AD
    if classification == 0 and confidence >= THRESHOLD_AD:
        return Label[0], Confidence[0]
    
    # Else proceed to OOD module
    net_ood = net_ood.cuda()
    net_ood.eval()
    pred, ood_score, confidence_ood = postprocessor.postprocess(net_ood, image)
    print(f'OOD Score: {ood_score}')
    # higher confidence means iD
    if ood_score <= THRESHOLD_BEST:
        classification = 2 
    print(f'OOD model: {classification}, {confidence}')

    if classification == 2 and ood_score < THRESHOLD_OOD:
        conf = 0
    if classification == 1 and ood_score > THRESHOLD_ID and confidence > THRESHOLD_HP:
        conf = 0
    print(Label[0])
    return Label[int(classification)], Confidence[conf]
