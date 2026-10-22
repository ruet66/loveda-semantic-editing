"""Portable versions of the actual semantic_04000 GAN and frozen helpers."""
import torch
from torch import nn
from torch.nn import functional as F

class Down(nn.Sequential):
    def __init__(self,a,b,norm=True):
        layers=[nn.Conv2d(a,b,4,2,1,bias=not norm)]
        if norm:layers.append(nn.InstanceNorm2d(b))
        super().__init__(*layers,nn.LeakyReLU(.2,inplace=False))

class Generator(nn.Module):
    def __init__(self):
        super().__init__();chs=[11,32,64,128,256,256,256]
        self.down=nn.ModuleList([Down(chs[i],chs[i+1],i!=0) for i in range(6)])
        self.up=nn.ModuleList([nn.Sequential(nn.ConvTranspose2d(a,b,4,2,1,bias=False),nn.InstanceNorm2d(b),nn.ReLU())
            for a,b in zip([256,512,512,256,128],[256,256,128,64,32])])
        self.out=nn.Sequential(nn.ConvTranspose2d(64,3,4,2,1),nn.Tanh())
        self.semantic_bias=nn.ModuleList([nn.Conv2d(7,c,1,bias=False) for c in [256,256,128,64,32]])
        for layer in self.semantic_bias:nn.init.zeros_(layer.weight)
    def forward(self,x):
        assert x.shape[1]==11
        sem=x[:,3:10];skips=[]
        for block in self.down:x=block(x);skips.append(x)
        for i,block in enumerate(self.up):
            x=block(x)+self.semantic_bias[i](F.interpolate(sem,size=(x.shape[-2]*2,x.shape[-1]*2),mode='nearest'))
            x=torch.cat([x,skips[-2-i]],1)
        return self.out(x)

class Discriminator(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Conv2d(14,32,4,2,1),nn.LeakyReLU(.2),Down(32,64),Down(64,128),
            nn.Conv2d(128,256,4,1,1,bias=False),nn.InstanceNorm2d(256),nn.LeakyReLU(.2),nn.Conv2d(256,1,4,1,1))

def onehot(labels):
    return torch.stack([labels==k for k in range(1,8)],1).float()

def condition(image,semantic,mask):
    return torch.cat([image*(1-mask),semantic,mask],1)

def normalize(image):
    mean=image.new_tensor([.485,.456,.406])[None,:,None,None]
    std=image.new_tensor([.229,.224,.225])[None,:,None,None]
    return ((image+1)/2-mean)/std

def segmentation(checkpoint,device):
    from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
    net=deeplabv3_mobilenet_v3_large(weights=None,weights_backbone=None,num_classes=7,aux_loss=False)
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    net.load_state_dict(state['model'],strict=True)
    return net.to(device).eval().requires_grad_(False)

def region_mean(values,mask):
    if values.shape[-2:]!=mask.shape[-2:]:mask=F.adaptive_avg_pool2d(mask,values.shape[-2:])
    denom=(mask.sum((1,2,3))*values.shape[1]).clamp_min(1)
    return ((values*mask).sum((1,2,3))/denom).mean()

def discriminator_features(net,x):
    features=[]
    for i,layer in enumerate(net):
        x=layer(x)
        if i in [1,2,3,6]:features.append(x)
    return x,features

class FrozenFeatures(nn.Module):
    def __init__(self,checkpoint):
        super().__init__()
        from torchvision.models import squeezenet1_1
        self.net=squeezenet1_1(weights=None).features[:8]
        self.net.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model'],strict=True)
        self.requires_grad_(False)
    def forward(self,x):
        values=[];x=normalize(x)
        for i,layer in enumerate(self.net):
            x=layer(x)
            if i in [1,4,7]:values.append(x)
        return values

def feature_losses(fake,real,mask):
    perceptual=mask.sum()*0;texture=perceptual
    for a,b in zip(fake,real):
        a=a.float();b=b.detach().float();m=F.adaptive_avg_pool2d(mask,a.shape[-2:])
        scale=b.square().mean((1,2,3),keepdim=True).sqrt().clamp_min(.1)
        a=a/scale;b=b/scale
        perceptual=perceptual+region_mean((a-b).abs(),m)/len(fake)
        denom=m.sum((2,3)).clamp_min(1)
        ma=(a*m).sum((2,3))/denom;mb=(b*m).sum((2,3))/denom
        va=((a-ma[:,:,None,None]).square()*m).sum((2,3))/denom
        vb=((b-mb[:,:,None,None]).square()*m).sum((2,3))/denom
        texture=texture+((ma-mb).abs().mean()+(torch.sqrt(va+1e-5)-torch.sqrt(vb+1e-5)).abs().mean())/len(fake)
    return perceptual,texture

def generator_loss(g,d,teacher,features,image,label,mask,iteration):
    semantic=onehot(label);cond=condition(image,semantic,mask)
    raw=g(cond);fake=image*(1-mask)+raw*mask
    valid=(label!=0)[:,None].float();support=mask*valid
    score,df=discriminator_features(d,torch.cat([cond,fake],1))
    adv=region_mean((score.float()-1).square(),support)
    l1=region_mean((raw.float()-image).abs(),support)
    ring=(mask+F.max_pool2d(-mask,7,1,3)).clamp_min(0)*valid
    boundary=region_mean((raw.float()-image).abs(),ring)
    with torch.no_grad():
        _,dr=discriminator_features(d,torch.cat([cond,image],1));real_features=features(image)
        probabilities=teacher(normalize(image))['out'].float().softmax(1)
        confidence,prediction=probabilities.max(1)
    fm=sum(region_mean((a.float()-b.float()).abs(),support) for a,b in zip(df,dr))/len(df)
    perceptual,texture=feature_losses(features(fake),real_features,support)
    gradient=sum(region_mean((torch.diff(fake.float(),dim=axis)-torch.diff(image,dim=axis)).abs(),
        support[:,:,:,1:] if axis==-1 else support[:,:,1:,:]) for axis in [-1,-2])/2
    target=label.long()-1;target=target.masked_fill(label==0,255)
    allowed=(prediction==target)&(confidence>=.5)&(label>=2)&(mask[:,0]>0)
    logits=teacher(normalize(fake))['out'].float()
    ce=region_mean(F.cross_entropy(logits,target,reduction='none',ignore_index=255)[:,None],allowed[:,None].float())
    loss=50*l1+10*boundary+2*adv+3*fm+2*perceptual+.5*texture+5*gradient+3*min(1.,iteration/500)*ce
    return loss,dict(l1=l1,boundary=boundary,adversarial=adv,feature_matching=fm,perceptual=perceptual,
        texture=texture,gradient=gradient,semantic_ce=ce,semantic_supervised_fraction=allowed.sum()/support.sum().clamp_min(1),
        semantic_allowed_pixels=torch.stack([(allowed&(label==k)).sum() for k in range(1,8)]))
