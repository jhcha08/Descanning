## Dataset Structure

1. Download our dataset DESCAN-18K from these links:  [scan1](https://drive.google.com/file/d/1Uanl0NPtVxVOwGb3yzGviopW-j0Gktc6/view?usp=sharing), [scan2](https://drive.google.com/file/d/16DxzIizRdxzrul1T-dgoIzhDn9szFpvK/view?usp=sharing), [clean](https://drive.google.com/file/d/1uB8rFMOjokdYz2ynSPHnxqgqpOAEW707/view?usp=sharing), [validation and test](https://drive.google.com/file/d/12txQIib3ycHcl4f8DscziVtRdN0qZZw1/view?usp=sharing)

2. Merge the folders named "scan1" and "scan2" into a single folder named "scan".

3. Organize the training, validation, and test datasets as follows:

dataset/  
├──train/  
    ├── clean/  
    │   ├── image1.tif  
    │   ├── image2.tif  
    │   ├── ...  
    └── scan/  
        ├── image1.tif  
        ├── image2.tif  
        ├── ...  
├──valid/  
    ├── clean/  
    │   ├── image1.tif  
    │   ├── image2.tif  
    │   ├── ...  
    └── scan/  
        ├── image1.tif  
        ├── image2.tif  
        ├── ...  
├──test/  
    ├── clean/  
    │   ├── image1.tif  
    │   ├── image2.tif  
    │   ├── ...  
    └── scan/  
        ├── image1.tif  
        ├── image2.tif  
        ├── ...  
