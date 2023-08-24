import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from datasets.torchvision_datasets.open_world import VOC_COCO_CLASS_NAMES
num_cls = 30
region_features = torch.load('output_prototype/region_features.pth')
gt_labels = torch.load('output_prototype/gt_labels.pth')

sample_features = []
sample_labels = []
sample_points = 200
for cls_inds in range(num_cls):
    cls_features = region_features[gt_labels==cls_inds]

    sample_feature = cls_features[:sample_points,:]
    sample_features.append(sample_feature)
    
    sample_label = torch.ones(len(sample_feature),dtype=torch.int64)*cls_inds
    sample_labels.append(sample_label)

region_features = torch.cat(sample_features,dim=0)
gt_labels = torch.cat(sample_labels,dim=0)
   
    
#t-sne visualize
nd_features = np.array(region_features)
labels = np.array(gt_labels)
#build tsne
tsne = TSNE()
pca_50 = PCA(n_components=50)
pca_result_50 = pca_50.fit_transform(nd_features)
#reduce feature
reduced_features = tsne.fit_transform(pca_result_50)
plt.figure(figsize=(20,15))
plt.scatter(reduced_features[:,0],reduced_features[:,1],s=8,c=gt_labels,cmap='Spectral')
plt.gca().set_aspect('equal','datalim')
plt.colorbar(boundaries=np.arange(num_cls+1)-0.5).set_ticks(np.arange(num_cls))
plt.title('Visualizing CODA train_set through t-SNE', fontsize=24)
plt.savefig('output_prototype/scatter_{}.png'.format(sample_points))
