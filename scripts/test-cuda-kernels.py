import torch
import pufferlib.native as nativelib

R = 20
C = 30
CR = 10
output = torch.zeros(R, 30).cuda()
nativelib.launch_linear_forward(torch.zeros(R, CR).cuda(),
                                torch.ones(C, CR).cuda(),
                                torch.zeros(R, C).cuda(),
                                output
                                )
 
print(f"Output : {output}")
