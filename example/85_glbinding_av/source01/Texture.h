#ifndef TEXTURE_H
#define TEXTURE_H


class Texture  {
        unsigned int image_texture;
        bool initialized_p;
        int m_internalFormat = 0;
        int m_width = 0;
        int m_height = 0;
        public:
        unsigned int GetImageTexture ()     ;  
        int GetWidth ()     ;  
        int GetHeight ()     ;  
        explicit  Texture (int w, int h, int internalFormat)     ;  
        void Update (unsigned char* data, int w, int h)     ;  
        bool Compatible_p (int w, int h, int internalFormat)     ;  
        void Reset (unsigned char* data, int w, int h, int internalFormat)     ;  
         ~Texture ()     ;  
};

#endif /* !TEXTURE_H */