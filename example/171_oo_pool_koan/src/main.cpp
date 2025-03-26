//
// Created by martin on 3/25/25.
//

#include "FixedSizeImagePool.h"
#include "GrayscaleImage.h"
#include "IImage.h"
#include "IImagePool.h"
#include <iostream>
int main() {
    FixedSizeImagePool pool(3,128,64);
    for (int i = 0; i < 100; i++) {
        IImage* image = pool.acquireImage();
        cout<<"Acqured image: "<<i<<endl;
        pool.releaseImage(image);
        cout <<"Released image: "<<i<<endl;
    }

    auto a = make_unique<int[]>(30);
    auto b = new int[10](2);
    cout << b[0] << endl;
    delete [] b;
    return 0;
}
