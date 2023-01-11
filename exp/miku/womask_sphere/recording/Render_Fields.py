import torch
import numpy as np
import os
import torch.optim as optim
from NeuS_src import xxx

"""
Generate a framework to run it first and then try to switch it to traditional format
"""

def Initial_Fields(Field_Shape=[64,64,64], Method='Traditional'):
    if Method == 'Traditional':
        SDF = torch.from_numpy(np.random.random(Field_Shape)).float()
        SDF.requires_grad = True
        CF = torch.from_numpy(np.zeros(Field_Shape)).float()
        CF.requires_grad = True
    elif Method == 'Network':
        SDF = torch.randn(Field_Shape)
        CF = torch.randn(Field_Shape)
    return SDF, CF

class Render_Fields(torch.nn.Module):
    def __init__(self, out_dir, Field_Shape=[64,64,64], Method='Traditional'):
        super().__init__()
        self.out_dir = 'Result/' + out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "mesh_Field"), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, "images_Field"), exist_ok=True)

        if Method == 'Traditional':
            SDF = torch.from_numpy(np.random.random(Field_Shape)).float()
            # self.SDF.requires_grad = True
            CF = torch.from_numpy(np.zeros(Field_Shape)).float()
            # self.CF.requires_grad = True
        elif Method == 'Network':
            SDF = torch.randn(Field_Shape)
            CF = torch.randn(Field_Shape)

        self.SDF = torch.nn.Parameter(SDF)
        self.CF = torch.nn.Parameter(CF)


    def forward(self, mvp, campos, resolution):

        # image = torch.randn([mvp.shape[0],resolution,resolution,3])
        # print(image)
        # image.requires_grad = True
        image = 2*self.SDF+self.CF
        # image.requires_grad = True
        return image

    def ptsd(self):
        return self.SDF, self.CF







if __name__ == "__main__":
    Field_Shape = [10,60, 60, 60]
    SS, CF = Initial_Fields(Field_Shape)
    Render_F=Render_Fields('F16',Field_Shape)
    mvp = torch.randn([10,4,4])


    Image_Ref = torch.randn([10,60,60,60])
    optimizer_Adam = optim.Adam(Render_F.parameters(), lr=0.001)
    SDF, CF = Render_F.ptsd()
    print(2*SDF[0]+CF[0]-Image_Ref[0])
    for i in range(3000):
        SSP = torch.clone(SS)
        Image_Field = Render_F(mvp, 2, 3)
        print(i)
        optimizer_Adam.zero_grad()
        Loss = torch.nn.L1Loss()(Image_Ref, Image_Field)
        print(Loss)
        Loss.backward()
        optimizer_Adam.step()
    SDF, CF = Render_F.ptsd()
    print(2*SDF[0]+CF[0]-Image_Ref[0])