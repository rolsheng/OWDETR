import torch
# model = torch.load('exps/r50_deformable_detr-checkpoint.pth')
# print(model.keys())
# model['epoch'] = 0
# # torch.save(model,'exps/checkpoint0000.pth')
# # class_embed.0.weight
model = torch.load("exps/checkpoint0169.pth")
format_weight = "class_embed.{}.weight"
format_bias = "class_embed.{}.bias"
print(model.keys())
model['epoch'] = 0
for i in range(6):
    model['model'][format_weight.format(i)] = model['model'][format_weight.format(i)][:31,:]
    model['model'][format_bias.format(i)] = model['model'][format_bias.format(i)][:31]
torch.save(model,"exps/pretrained_weight.pth")
