import torch.nn as nn
import torch 



class UNet(nn.Module):

    """
    UNet architecture, with hyperparameter depth
    """

    def __init__(self, num_classes=4, in_channels=1, init_weights=True, depth=4):
        """
        
        """
        super(UNet, self).__init__()
        # encoder

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = num_classes

        size=64

        # couches de la descente de UNet
        self.down_layers = nn.ModuleList(

            # la première couche a une taille d'entrée différente
            [
            nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=size, kernel_size=3, stride=1, padding="same"),
                nn.BatchNorm2d(size),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=size, out_channels=size, kernel_size=3, stride=1, padding="same"),
                nn.BatchNorm2d(size),
                nn.ReLU(inplace=True),
            )

            # les autres couches suivent le même schéma:
            ] + [ 
            
            nn.Sequential(
                nn.Conv2d(in_channels=size*(2**i), out_channels=size*(2**(i+1)), kernel_size=3, stride=1, padding="same"),
                nn.BatchNorm2d(size*(2**(i+1))),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=size*(2**(i+1)), out_channels=size*(2**(i+1)), kernel_size=3, stride=1, padding="same"),
                nn.BatchNorm2d(size*(2**(i+1))),
                nn.ReLU(inplace=True),
            )

            for i in range(0,depth-1)


        ])

        # la couche du milieu est elle aussi un peu différente:
        self.mid_layer=nn.Sequential(
                nn.Conv2d(in_channels=size*(2**depth), out_channels=size*(2**(depth+1)), kernel_size=3, stride=1, padding="same"),
                nn.BatchNorm2d(size*(2**(depth+1))),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=size*(2**(depth+1)), out_channels=size*(2**depth), kernel_size=3, stride=1, padding="same"),
                nn.BatchNorm2d(size*(2**depth)),
                nn.ReLU(inplace=True),
            )

        # couches de la remonté de UNet
        self.up_layers = nn.ModuleList(
            
            #couches classiques
            [
            nn.Sequential(
                nn.Conv2d(in_channels=size*(2**i), out_channels=size*(2**(i-1)), kernel_size=3, stride=1, padding="same"),
                nn.BatchNorm2d(size*(2**(i-1))),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=size*(2**(i-1)), out_channels=size*(2**(i-2)), kernel_size=3, stride=1, padding="same"),
                nn.BatchNorm2d(size*(2**(i-2))),
                nn.ReLU(inplace=True),
            )
            for i in range(depth,1,-1)

            #dernière couche un peu spécial
            ]+[

            nn.Sequential(
                nn.Conv2d(in_channels=size*2, out_channels=size, kernel_size=3, stride=1, padding="same"),
                nn.BatchNorm2d(size),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=size, out_channels=size, kernel_size=3, stride=1, padding="same"),
                nn.BatchNorm2d(size),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=size, out_channels=num_classes, kernel_size=3, stride=1, padding="same"),
            )
        ])

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.max_unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

    def forward(self, x):
        """
        Forward pass of the model
        Args:
            x: Tensor
        """
        outputs = [x] # remember output sizes for unpooling and concatenate
        indices = [] # remember pooling indices for the unpooling

        prev_layer = x     
        for layer in self.down_layers:
            out = layer(prev_layer) # convolutions 
            prev_layer, ind = self.max_pool(out) # pooling
            indices.append(ind) # save in stacks
            outputs.append(out)

        for layer in self.up_layers:
            last_output=outputs.pop()
            out = self.max_unpool(prev_layer, indices.pop(), output_size=last_output.size()) # unpooling
            out = torch.cat((out,last_output),dim=1) #concatenate
            prev_layer = layer(out) # convolutions

        return None, None, prev_layer