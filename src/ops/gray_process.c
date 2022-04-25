#include "gray_process.h"

#ifdef GRAY

Tensor *img_reversal(Tensor *img)
{
    Tensor *new_img = copy_image(img);
    for (int i = 0; i < img->num; ++i){
        new_img->data[i] = 1 - new_img->data[i];
    }
    return new_img;
}

Tensor *Log_transfer(Tensor *img, float c)
{
    Tensor *new_img = copy_image(img);
    for (int i = 0; i < img->num; ++i){
        new_img->data[i] = c * log(new_img->data[i]+1/255.);
    }
    return new_img;
}

Tensor *power_law(Tensor *img, float c, float gamma, float varepsilon)
{
    Tensor *new_img = copy_image(img);
    for (int i = 0; i < img->num; ++i){
        new_img->data[i] = c * pow((new_img->data[i] + varepsilon), gamma);
    }
    return new_img;
}

Tensor *contrast_stretch(Tensor *img, float r_max, float r_min, float s_max, float s_min)
{
    Tensor *new_img = copy_image(img);
    for (int i = 0; i < img->num; ++i){
        new_img->data[i] = (new_img->data[i] - r_min) / (r_max - r_min) * (s_max - s_min) + s_min;
    }
    return new_img;
}

Tensor *gray_slice_1(Tensor *img, float slice, float focus, float no_focus)
{
    Tensor *new_img = copy_image(img);
    for (int i = 0; i < img->num; ++i){
        if (new_img->data[i] >= slice) new_img->data[i] = focus;
        else new_img->data[i] = no_focus;
    }
    return new_img;
}

Tensor *gray_slice_2(Tensor *img, float slice, int flag, float focus)
{
    Tensor *new_img = copy_image(img);
    for (int i = 0; i < img->num; ++i){
        if (flag){
            if (new_img->data[i] >= slice) new_img->data[i] = focus;
        }
        else{
            if (new_img->data[i] <= slice) new_img->data[i] = focus;
        }
    }
    return new_img;
}

Tensor **bit8_slice(Tensor *img)
{
    Tensor **new_imgs = malloc(8*sizeof(Image*));
    for (int i = 0; i < 8; ++i){
        new_imgs[i] = create_image(img->size[0], img->size[1], img->size[2]);
    }
    for (int i = 0; i < img->num; ++i){
        binary *bits = int2binary((int)(img->data[i]*255));
        node *n = bits->tail;
        for (int j = 0; j < 8; ++j){
            if (n){
                new_imgs[j]->data[i] = n->data;
                n = n->prev;
            }
            else new_imgs[j]->data[i] = 0;
        }
    }
    return new_imgs;
}

Tensor *bit_restructure(Tensor **imgs, int *bit_index, int len)
{
    Tensor *img = create_image(imgs[0]->size[0], imgs[0]->size[1], imgs[0]->size[2]);
    for (int i = 0; i < len; ++i){
        Tensor *bit_img = imgs[i];
        for (int j = 0; j < img->num; ++j){
            img->data[j] += pow(2, bit_index[i]) * bit_img->data[j] / 255.;
        }
    }
    return img;
}

Tensor *histogram_equalization(Tensor *img)
{
    Tensor *new_img = create_image(img->size[0], img->size[1], img->size[2]);
    for (int i = 0; i < img->size[2]; ++i){
        __histogram_equalization_channel(img, new_img, i+1);
    }
    return new_img;
}

void __histogram_equalization_channel(Tensor *o_img, Tensor *a_img, int c)
{
    int p_num = o_img->size[0] * o_img->size[1];
    int *num = census_channel_pixel(o_img, c);
    int *gray_level = calloc(256, sizeof(int));
    for (int i = 0; i < 256; ++i){
        int n = 0;
        for (int j = 0; j <= i; ++j){
            n += num[j];
        }
        gray_level[i] = 255. / p_num * n;
    }
    int offset = (c - 1) * p_num;
    for (int i = 0; i < p_num; ++i){
        a_img->data[i+offset] = gray_level[(int)(o_img->data[i+offset]*255)] / 255.;
    }
    free(num);
    free(gray_level);
}

#endif