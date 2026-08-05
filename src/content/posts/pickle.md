---
title: Store Almost Any Objects of Python in Files 
date: 2020-04-15T15:50:20-07:00
tags: [python, package]
categories: NOTE
description: Cache outputs of time consuming code results
---

The module [`pickle`](https://docs.python.org/3/library/pickle.html) implements binary protocols for serializing and de-serializing a Python object structure. We can store almost any type of object of Python using `pickle`.



## Quick Example

For example, we want to save a dictionary `dict_obj` to file.

```python
# Save file
def saveFile(obj):
    out_file = open('obj.pickle','wb')
    pickle.dump(obj, out_file)
    out_file.close()
    
# Read file
def readFile(obj_file_name):
    file = open(obj_file_name, 'rb')
    obj = pickle.load(file)
    return obj

>>> dict_obj = {'itemA': ['item', 'A'], 'itemB':[1, 3]}
>>> saveFile(dict_obj)
>>> obj_file_name = 'obj.pickle'
>>> readFile(obj_file_name)
{'itemA': ['item', 'A'], 'itemB':[1, 3]}
```



## Related Package: JSON

#### Comparison of pickle and JSON

There are some fundamental differences between the pickle protocols and JSON.

> - JSON is a text serialization format (it outputs Unicode text, although most of the time it is then encoded to `utf-8`), while pickle is a binary serialization format;
>
> - JSON is human-readable, while pickle is not.
>
> - JSON is widely used outside of Python ecosystem, while pickle is Python-specific.
>
> - JSON, by default, **can only represent a subset of the Python built-in types**, and no custom classes; pickle can represent an extremely large number of Python types.

\- [pickle — Python object serialization](https://docs.python.org/3/library/pickle.html)

![The same objects written out by pickle and by json: an opaque byte stream that keeps the class, versus readable text that does not](/img/pickle/pickle-vs-json.svg)

The trade is the same object leaving memory by two different doors. `json.dump` produces something you can open in any editor and read on any platform, at the cost of flattening everything down to a handful of built-in types. `pickle.dump` produces bytes only Python can interpret, and in exchange the object comes back as the class it was.

#### JSON Example

Note that JSON can only store a subset of object of Python. Dictionary is one of them.

```python
import json

# Store dictionary
>>> with open('obj.json', 'wb') as file:
...     json.dump(dict_obj, file)
    
# Read in dictionary file
>>> file = open('obj.json', 'rb')
>>> obj = json.load(file)
>>> obj
{'itemA': ['item', 'A'], 'itemB':[1, 3]}
```



## Suggestions on Choosing Pickle and JSON

As we can see, pickle and JSON nearly shared the same syntax and complexity in terms of storing and loading files. The major differences are 

1. Whether do we need readable file format;
2. Whether the object we want to save is a python build-in object;

If the answers are ‘Yes’ to both questions, we should choose JSON. Otherwise, pickle would be the better option here.



