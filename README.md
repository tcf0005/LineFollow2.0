# LineFollow2.0
Updated Version of 2021's Soft Sys Reccomended Vision Algorithm 
## Added:
* Sobel Derivative Filter to Find Edges
* Removed HSV Filter 
  * Replaced with Saturation Filter (Channel of HSL)
* Sliding Box Algorithm


## To Compile and Run:
```bash
git clone https://github.com/tcf0005/LineFollow2.0.git
cd LineFollow2.0
cmake .
make 
./LF2
```

## Algorithm Steps
1. Sobel Operator Applied to Grayscale Input Image
2. Saturation Determined by converting input image into HSL Colorspace
3. Steps 1 and 2 are combined into a single image
4. Histogram Determined to Find X-Coordinate with Maximum Non-Zero Pixels
5. Sliding Box Detector Applied to Discretize the Line (Intialized Using Step 4)

## Useful Links
### Actual Lane Following Algorithm This was Based on: 
https://navoshta.com/detecting-road-features/
