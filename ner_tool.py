
!pip install git+https://github.com/hiendang7613/FPromptify.git
!pip3 install openai huggingface_hub

import json
import time
from collections import defaultdict

from promptify import OpenAI
from promptify import Prompter
 

def __main__():

  start = time.time()
  sentence     =  '''Tôi là Nguyễn Thái Học và Đặng Văn Hiển và làm việc tại FujiNet, tôi sống tại Phú Tài Building (tầng 4), 278 Nguyễn Thị Định, Thành Phố Quy Nhơn, Tỉnh Bình Định.

  Điện thoại: (84-28) 3847-7000

  Fax: (84-28) 3847-5000  

  https://colab.research.google.com/drive/1ojN
  '''
  loop = True

  while (loop):

    try:
      model = OpenAI('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') # or `HubModel()` for Huggingface-based inference
      loop = False
      time.sleep(0.5)

    except Exception as e: 
      print(e)
      loop = True

  print(model)
  nlp_prompter = Prompter(model)

  labels = ', '.join([
      'Telephone number',
      'Fax',
      'Mobile',
      'Email',
      'website/url',
      'People',
      'Company',
      'Department',
      'Position',
      'Address',
  ])

  result       = nlp_prompter.fit('ner.jinja',
                            domain      = 'company information',
                            text_input  = sentence, 
                            labels      = labels)

  end = time.time()
  print(end-start)
  return result['text']

__main__()