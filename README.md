# azure-function-accent-scoring

## Overview

An Azure function that accepts HTTP request with mp3 blob string, decodes mp3 and predicts an accent score with the model stored in Azure Storage.

### HTTP request format

//TODO

### HTTP response format

//TODO

## General Info

General information of this Azure Function can be accessed from `Home > Function App > accent-scoring`

## Updating accent scoring model

1.   Open Azure Portal

2.   Navigate to `Home > Storage Accounts > accentscoring > File shares`

3.   Upload the model.pt file in the `model/` directory

## Accessing model in Azure File Share for another Azure Function
```
az webapp config storage-account add 
    --resource-group cockatoos
    --name $nameOfAzureFunction
    --custom-id $randomString 
    --storage-type AzureFiles
    --share-name model
    --account-name accentscoring
    --mount-path /model
    --access-key $storageAccountKey
```
* storageAccountKey can be accessed from Azure Portal `Home > Storage Accounts > accentscoring > Access Keys`

## Create a new Azure File Share to be accessed by Azure Functions
```
az storage share create
    --account-name accentscoring
    --account-key $storageAccountKey
    --name $nameOfNewDir
    --quota 1024
```

* storageAccountKey can be accessed from Azure Portal `Home > Storage Accounts > accentscoring > Access Keys`
