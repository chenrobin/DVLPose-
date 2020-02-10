class LSTM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstmcell = torch.nn.LSTM(
            inpout_size = 256,
            hidden_size = 256,
            num_layers = 1,
            batch_first = True
        )
    def forward(self,x):
        output,(h_n,c_n) = self.lstmcell(x)
        # output_last = h_n[-1,:,:]
        return output,(h_n,c_n)

class getPose(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.final_layer = nn.Conv2d(
            in_channels=256,
            out_channels=17,
            kernel_size=1,
            stride=1,
            padding=0
        )
    def forward(self,x):
        x = self.final_layer(x)
        return x
