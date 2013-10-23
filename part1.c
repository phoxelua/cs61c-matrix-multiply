#include <emmintrin.h>
#define KERNX 3 //this is the x-size of the kernel. It will always be odd.
#define KERNY 3 //this is the y-size of the kernel. It will always be odd.
int conv2D(float* in, float* out, int data_size_X, int data_size_Y,
                    float* kernel)
{
    // the x coordinate of the kernel's center
    int kern_cent_X = (KERNX - 1)/2;
    // the y coordinate of the kernel's center
    int kern_cent_Y = (KERNY - 1)/2;
    
    // main convolution loop
	for(int x = 0; x < data_size_X; x++){ // the x coordinate of the output location we're focusing on
		for(int i = -kern_cent_X; i <= kern_cent_X; i++){ // kernel unflipped x coordinate
			for(int y = 0; y < data_size_Y; y++){ // the y coordinate of theoutput location we're focusing on
				for(int j = -kern_cent_Y; j <= kern_cent_Y; j++){ // kernel unflipped y coordinate
					// only do the operation if not out of bounds
					if(x+i>-1 && x+i<data_size_X && y+j>-1 && y+j<data_size_Y){
						//Note that the kernel is flipped
						//__m128i in = _mm_setzero_si128();
					        //__m128i sum = _mm_setzero_si128();
						__m128 kern =  _mm_load_ps(kernel + ((kern_cent_X-i)+(kern_cent_Y-j)*KERNX));
						__m128 in_temp =  _mm_loadu_ps(in + ((x+i) + (y+j)*data_size_X));
						__m128 out_temp = _mm_loadu_ps(kernel + (x+y*data_size_X));
						float final  = 0;
						//__m128i noreason = _mm_setzero_si128();
						kern = _mm_mul_ps(kern,in_temp);
						out_temp = _mm_add_ps(kern, out_temp);
						for (int a=0; a<4;a++){
							final+= *(out+a);
						}
						out[x+y*data_size_X] = final;
						//		kernel[(kern_cent_X-i)+(kern_cent_Y-j)*KERNX] * in[(x+i) + (y+j)*data_size_X]
					}
				}
			}
		}
	}
	return 1;
}
