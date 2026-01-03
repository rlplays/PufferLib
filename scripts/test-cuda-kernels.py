import torch
import pufferlib.native as nativelib

output = torch.zeros(10, 30).cuda()
stream = torch.streams.cuda.current()
nativelib.launch_linear_forward(torch.zeros(10, 20).cuda(),
                                torch.zeros(30, 20).cuda(),
                                torch.zeros(10, 30).cuda(),
                                output,
                                stream
                                )
 
print(f"Output : {output}")
