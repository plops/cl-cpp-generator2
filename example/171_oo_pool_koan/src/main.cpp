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
    cout << b[0] << endl; // 2
    cout << b[1] << endl; // 0 should be 2
    delete [] b;

    auto c = new GrayscaleImage[5](12,13);
    cout << c[0].getHeight() << endl; //  this prints 64 (i want it to print 13)
    delete [] c;

    return 0;
}
