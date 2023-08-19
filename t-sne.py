import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from datasets.torchvision_datasets.open_world import VOC_COCO_CLASS_NAMES
num_cls = 30
points = [10,50,100,200,500]
region_features = torch.load('output_prototype/region_features.pth')
gt_labels = torch.load('output_prototype/gt_labels.pth')
assert len(region_features)==len(gt_labels)
#t-sne visualize
nd_features = np.array(region_features.clone())
labels = np.array(gt_labels)
tsne = TSNE(init="pca",learning_rate='auto')
#sampling
colors = plt.cm.jet(np.linspace(0, 1, num_cls))
for point in points:
    handles = []
    tags = [] 
    plt.figure(figsize=(20,15))
    for cls_inds in range(num_cls):
        cls_features = nd_features[labels==cls_inds]
        num_samples = len(cls_features)
        random_indices = np.random.choice(num_samples,size=min(num_samples,point),replace=False)
        sample_features = cls_features[random_indices]
        reduced_features = tsne.fit_transform(sample_features)
        scatter = plt.scatter(reduced_features[:,0],reduced_features[:,1],c = colors[cls_inds],alpha=0.5)
        handles.append(scatter)
        tags.append(VOC_COCO_CLASS_NAMES[cls_inds])
    plt.axis('off')
    plt.legend(handles,tags,loc='upper right',bbox_to_anchor=(2,2))
    plt.savefig('output_prototype/scatter_{}.png'.format(point))
    # plt.title(label="{} sampled points".format(point))