import tritonclient.http as tritonhttpclient
import tritonclient.grpc as tritongrpcclient
import numpy as np
from PIL import Image
from torchvision import transforms
import json

###CONFIGURATION########
VERBOSE = False
input_name = 'actual_input_1'
input_shape = (1, 3, 224, 224)
input_dtype = 'FP32'
output_name = 'output_1'
model_name = 'trt_model'
http_url = 'localhost:8000'
grpc_url = 'localhost:8001'
model_version = '1'
########################

#Image Loading
image = Image.open('./src/goldfish.jpg')

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.485, 0.456, 0.406]

resize = transforms.Resize((256, 256))
center_crop = transforms.CenterCrop(224)
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=imagenet_mean,
                                 std=imagenet_std)

transform = transforms.Compose([resize, center_crop, to_tensor, normalize])
image_tensor = transform(image).unsqueeze(0).cuda()

#Label Loading

with open('./src/imagenet-simple-labels.json') as file:
    labels = json.load(file)

#Start client set up

#triton_client = tritonhttpclient.InferenceServerClient(url=http_url, verbose=VERBOSE)
triton_client = tritongrpcclient.InferenceServerClient(url=grpc_url, verbose=VERBOSE)
model_metadata = triton_client.get_model_metadata(model_name=model_name, model_version=model_version) #You can remove this line
model_config = triton_client.get_model_config(model_name=model_name, model_version=model_version)

image_numpy = image_tensor.cpu().numpy()
print(image_numpy.shape)

#input0 = tritonhttpclient.InferInput(input_name, input_shape, input_dtype)
input0 = tritongrpcclient.InferInput(input_name, input_shape, input_dtype)
#input0.set_data_from_numpy(image_numpy, binary_data=False)
input0.set_data_from_numpy(image_numpy)

#output = tritonhttpclient.InferRequestedOutput(output_name, binary_data=False)
output = tritongrpcclient.InferRequestedOutput(output_name)
response = triton_client.infer(model_name, model_version=model_version, 
                               inputs=[input0], outputs=[output])
logits = response.as_numpy(output_name)
logits = np.asarray(logits, dtype=np.float32)

print(labels[np.argmax(logits)])
