import ollama
from ollama import Client

host = 'https://select-indirectly-jennet.ngrok-free.app'
# host = 'http://localhost:11434'
model = 'llama3.2'
try:
  ollama.chat(model)
except ollama.ResponseError as e:
  print('Error:', e.error)
  if e.status_code == 404:
    ollama.pull(model)

client = Client(
  host=host#,
#   headers={'x-some-header': 'some-value'}
)
response = client.chat(model=model, messages=[
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])

print(response.message.content)
#   """model='llama3.2' created_at='2024-12-05T12:43:07.377401807Z' done=True done_reason='stop' total_duration=5215300855 load_duration=40790113 prompt_eval_count=31 prompt_eval_duration=55000000 eval_count=323 eval_duration=5118000000 message=Message(role='assistant', content="The sky appears blue to us due to a phenomenon called Rayleigh scattering, named after the British physicist Lord Rayleigh, who first described it in the late 19th century.\n\nHere's what happens:\n\n1. **Sunlight enters Earth's atmosphere**: When sunlight enters our atmosphere, it encounters tiny molecules of gases such as nitrogen (N2) and oxygen (O2).\n2. **Scattering occurs**: These molecules scatter the light in all directions, but they scatter shorter (blue) wavelengths more than longer (red) wavelengths.\n3. **Blue light is scattered**: The blue light is dispersed in all directions by the tiny molecules, which is why it reaches our eyes from all parts of the sky.\n4. **Red light continues straight**: Meanwhile, the longer wavelength red light continues to travel in a straight line, reaching our eyes directly from the sun's position in the sky.\n\nThis scattering effect is more pronounced for shorter wavelengths because they have more energy and are more easily deflected by the tiny molecules. As a result, our eyes perceive the blue color of the sky because that's the wavelength that's scattered most.\n\nInterestingly, during sunrise and sunset, the sky can appear red or orange due to another phenomenon called Mie scattering. This occurs when light interacts with larger particles in the atmosphere, such as dust and water droplets, which scatter longer wavelengths more than shorter 
# ones.\n\nIn summary, the sky appears blue because of Rayleigh scattering, where tiny molecules in the atmosphere scatter shorter wavelengths (blue light) more than longer wavelengths (red light), resulting in our perception of a blue color.", images=None, tool_calls=None)
#   """