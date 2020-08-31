# DMA Regularization
Code for "DMA Regularization: Enhancing Discriminability of Neural Networks by Decreasing the Minimal Angle". 

The code is implemented based on Pytorch.

The code is based on https://github.com/bearpaw/pytorch-classification.git, a widely used classification Repo including many modern neural network models. In the following, we just illustrate the contribution of our code.

# Usage

Firstly, the classification layer of model needs to be replaced with DMA_Linear. For example, in the ResNet model:
        
        import DMA
        
        self.classifier = DMA.DMA_Linear(64 * block.expansion, num_classes)

Secondly, in the training and test code, the output of model has 2 elements:
        
        outputs, cosine = model(inputs, targets)

Thirdly, use DMA regularization in trainging procedure:
        
        from DMA import dma_loss

        
        # in training method 
        ...
        ...     
           
        # normal learning loss
        outputs, cosine = model(inputs, targets)
        loss = criterion(outputs, targets)
        
        # DMA regularization
        dmaloss = dma_loss(cosine)
        loss += coefficient * dmaloss

        loss.backward() 
 
# To realize deterministic training, following code is needed:

        import torch
        import torch.backends.cudnn as cudnn
        import random
        
        cudnn.benchmark = False
        cudnn.deterministic = True
        
        manualSeed = 123
        
        random.seed(manualSeed)
        torch.manual_seed(manualSeed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(manualSeed)
            
        ...
        ...
            
        # still need to set the work_init_fn to random.seed in train_dataloader, if multi numworkers
        trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers, worker_init_fn=random.seed)
        
# License
This code is released under the MIT License (refer to the LICENSE file for details).

# Notes      
The usage in other deep learning library, is similar. And the default coefficient is set to 0.5. However, it may need to be tuned according to different task, model, and dataset.
