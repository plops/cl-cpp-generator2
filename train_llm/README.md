- this directory contains python code to collect the input
  s-expressions and corresponding output for all the examples in
  cl-cpp-generator2
  
- the data is collected in a json file of this format:

```
[
        {
             'text_input': '1',
             'output': '2',
        },{
             'text_input': '3',
             'output': '4',
        },{
...
}
]
```

- this will serve as finetuning input for a large language model

- the llm shall transform c++ input code to s-expressions 
