import torch
import pufferlib.native as nativelib

R = 128
C = 118
CR = 10
output = torch.zeros(R, C).cuda()
nativelib.launch_linear_forward(torch.ones(R, CR).cuda(),
                                torch.ones(C, CR).cuda(),
                                torch.zeros(R, C).cuda(),
                                output
                                )
 
print(f"Output : {output}")
