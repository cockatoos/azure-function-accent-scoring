<a name="top"></a>

# azure-function-accent-scoring [![Linter](https://github.com/cockatoos/azure-function-accent-scoring/actions/workflows/main.yml/badge.svg)](https://github.com/cockatoos/azure-function-accent-scoring/actions/workflows/main.yml)

## Overview

An Azure function runs in a Docker container that accepts HTTP request with mp3 blob string, decodes mp3 and predicts an accent score with the model .

### HTTP request format
```
{"blob": string}
```

### HTTP response format
```
{"status": "success", "score": string}
{"status": "failure", "reason": string}
```

## General Info

General information of this Azure Function can be accessed from `Home > Function App > accent-scoring`

## Updating accent scoring model

1.   Open Azure Portal

2.   Navigate to `Home > Storage Accounts > accentscoring > File shares`

3.   Upload the model.pt file in the `model/` directory
