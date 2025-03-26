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
    auto b = new int[10](2,2,2);
    cout << b[0] << endl; // 2
    cout << b[1] << endl; // 2
    cout << b[9] << endl; // 0 should be 2
    delete [] b;

    auto c = new GrayscaleImage[5]({12,13});
    cout << c[0].getHeight() << endl; // 13
    delete [] c;

    const int defaultWidth = 12;
    const int defaultHeight = 13;
    class DefaultGrayscaleImage : public GrayscaleImage {
        public:
        DefaultGrayscaleImage() :GrayscaleImage(defaultWidth,defaultHeight) {}
    };
    auto d = new DefaultGrayscaleImage[5];
    cout << d[0].getWidth() << endl; // 12
    cout << d[2].getWidth() << endl; // 12 .. this works
    delete [] d;

    return 0;
}
