#include <emmintrin.h>
#include <omp.h>
#define KERNX 3 //this is the x-size of the kernel. It will always be odd.
#define KERNY 3 //this is the y-size of the kernel. It will always be odd.
int conv2D(float* in, float* out, int data_size_X, int data_size_Y,
                    float* kernel)
{
// the x coordinate of the kernel's center
int kern_cent_X = (KERNX - 1)/2;
// the y coordinate of the kernel's center
int kern_cent_Y = (KERNY - 1)/2;

	
//pad with zeros
int x_padding = (4 - (data_size_X % 4)) % 4; //how much we have to pad the rows to be a multiple of 4
int new_size_X = data_size_X + 2 + x_padding;
int new_size_Y = data_size_Y + 2;
int new_size = new_size_X *new_size_Y;	

float newIn[new_size];

	int blocksize;
	if (data_size_X/100 == 4 && data_size_Y/100 == 4){
		blocksize = 50;
	}
	else if (data_size_X/100 == 4 && data_size_Y/100 == 6){
		blocksize = 20;
	}
	else if (data_size_X/100 == 6 && data_size_Y/100 == 4){
		blocksize = 50;
	}
	else if (data_size_X/100 == 12 && data_size_Y/100 == 12){
		blocksize = 150;
	}
	else{
		blocksize = 100;
	}

     int i, j, k,l;
     int a,b;
     int lnewx;
     int ldatax;
    #pragma omp parallel for private(i,j,k,l,lnewx,ldatax) num_threads(8)
     for (j=0; j<data_size_Y; j+=blocksize){ 
     	for (i=0; i<data_size_X; i+=blocksize){ 
         for(l=j; l<j+blocksize && l<data_size_Y; l++){
	     lnewx=(l+1)*new_size_X;
	     ldatax=l*data_size_X;
            for(k=i; k<i+blocksize && k<data_size_X; k+=4){
		_mm_storeu_ps(newIn + (k+1)+lnewx, _mm_loadu_ps(in + k + ldatax));
                    //newIn[(k+1)+(l+1)*new_size_X] = in[k+l*data_size_X];	  	
	     }
	  }
	}
     }	

     if (data_size_X % blocksize != 0){
	//printf("remainder loop X\n");
	int bnsx, bdsx; 
	int start = data_size_X - data_size_X%blocksize;
	#pragma omp parallel for private(a,b,bnsx,bdsx) num_threads(8)
       for(b = 0; b < data_size_Y; b++){
		bnsx = (b+1)*new_size_X;
		bdsx = b*data_size_X;
      		for(a = start; a < data_size_X; a+=4){ 
			_mm_storeu_ps(newIn + (a+1)+bnsx, _mm_loadu_ps(in + a + bdsx));
            		//newIn[(a+1) + (b+1)*new_size_X] = in[a + b*data_size_X];
	  	}
	}
     }
     if (data_size_Y % blocksize != 0){
	//printf("remainder loop Y\n");
	int bnsx, bdsx; 
	int start = data_size_Y - data_size_Y%blocksize;
	#pragma omp parallel for private(a,b,bnsx,bdsx) num_threads(8)
        for(b = start; b < data_size_Y; b++){
		bnsx = (b+1)*new_size_X;
		bdsx = b*data_size_X;
      		for(a = 0; a < data_size_X; a+=4){
			_mm_storeu_ps(newIn + (a+1)+bnsx, _mm_loadu_ps(in + a + bdsx));
            		//newIn[(a+1) + (b+1)*new_size_X] = in[a + b*data_size_X];
	  	}
	}
     }

	//Padds left/right cols with zeros
	float zeros[4] = {0,0,0,0};
	int nsxnsy = new_size_X*(new_size_Y-1);
	int end = new_size_X/4*4;
	 for (int i=0; i<end;){
		 _mm_storeu_ps(newIn + i, _mm_loadu_ps(zeros));
		 _mm_storeu_ps(newIn + i + nsxnsy, _mm_loadu_ps(zeros));
		i+=4;
	 }
	 
	//Pad leftover left/right  with zeros 
	#pragma omp parallel for private(i) num_threads(8) 
	for (int i=end; i<new_size_X;i++){
		*(newIn + i) = 0;
		*(newIn + i + nsxnsy) = 0;
		// newIn[i] = 0;
		// newIn[i + nsxnsy] = 0;
	 }

	 //Pad newIn top/bottom rows with zeros
	int nsxj,nsxj1;
	#pragma omp parallel for private(j,i) num_threads(8)
	 for (int j=0; j<new_size_Y;j++){
		 nsxj = new_size_X *j;
		 nsxj1 = new_size_X *(j+1) - 1;
		*(newIn + nsxj) = 0;
		*(newIn + nsxj1) = 0;
	 	for(int i=1;i<=x_padding; i++){
			*(newIn - i + nsxj1) = 0;
		}
	 }


	//Flip kernel
	__m128 newKern[9];
	newKern[0] = _mm_load1_ps(kernel + 8);
	newKern[1] = _mm_load1_ps(kernel  + 7);
	newKern[2] = _mm_load1_ps(kernel  + 6);
	newKern[3] = _mm_load1_ps(kernel  + 5);
	newKern[4] = _mm_load1_ps(kernel  + 4);
	newKern[5] = _mm_load1_ps(kernel  + 3);
	newKern[6] =_mm_load1_ps(kernel  + 2);
	newKern[7] = _mm_load1_ps(kernel  + 1);
	newKern[8] =  _mm_load1_ps(kernel);

	///MAIN CONVOLUTION LOOP////
	int x,y;
	__m128 out_temp, kern0, kern1, kern2, kern3, kern4, kern5, kern6, kern7, kern8, mult0, mult1, mult2, in_temp0, in_temp1, in_temp2; 
	//blocksize = 100;
	#pragma omp parallel for  private(y,x,out_temp, kern0, kern1, kern2, kern3, kern4, kern5, kern6, kern7, kern8, mult0, mult1, mult2, in_temp0, in_temp1, in_temp2) num_threads(8)
	//for (j=1;j<new_size_Y-1;j+=blocksize){
	//for (i=1;i<new_size_X-1;i+=blocksize){
	//for(y = j; y<j+blocksize && y < new_size_Y-1; y++){ // the y coordinate of theoutput location we're focusing on
	//	for(x = i; x<i+blocksize && x < new_size_X-1;x+=4){ // the x coordinate of the output location we're focusing on
	for(y = 0; y < data_size_Y; y++){ // the y coordinate of theoutput location we're focusing on
		for(x = 0; x < data_size_X/4*4;x+=4){ // the x coordinate of the output location we're focusing on
						//First
						out_temp = _mm_loadu_ps(out+ (x+y*data_size_X));
						kern0 =  *(newKern + 0);
						in_temp0 =  _mm_loadu_ps(newIn + x + (y+1-kern_cent_Y)*new_size_X);
						mult0 = _mm_mul_ps(in_temp0,kern0);
						out_temp = _mm_add_ps(out_temp, mult0);


						kern1 = *(newKern + 1);
						in_temp1 =  _mm_loadu_ps(newIn + ((x+1) + (y+1-kern_cent_Y)*new_size_X));
						mult1 = _mm_mul_ps(in_temp1,kern1);
						out_temp = _mm_add_ps(out_temp, mult1);

						kern2 = *(newKern  + 2);
						in_temp2 =  _mm_loadu_ps(newIn + ((x+2) + (y+1-kern_cent_Y)*new_size_X));
						mult2 = _mm_mul_ps(in_temp2, kern2);
						out_temp = _mm_add_ps(out_temp, mult2);


						//Sec
						kern3 = *(newKern + 3);
						in_temp0 = _mm_loadu_ps(newIn + ((x) + (y+1-kern_cent_Y+1)*new_size_X));
						mult0 = _mm_mul_ps(kern3, in_temp0);
						out_temp = _mm_add_ps(out_temp, mult0);

						kern4 = *(newKern  + 4);
						in_temp1 =  _mm_loadu_ps(newIn + ((x+1) + (y+1-kern_cent_Y+1)*new_size_X));
						mult1 = _mm_mul_ps(kern4,in_temp1);
						out_temp = _mm_add_ps(out_temp, mult1);

						kern5 =  *(newKern + 5);
						in_temp2 =  _mm_loadu_ps(newIn + ((x+2) + (y+1-kern_cent_Y+1)*new_size_X));
						mult2 = _mm_mul_ps(kern5,in_temp2);
						out_temp = _mm_add_ps(out_temp, mult2);


						//Third
						kern6 =  *(newKern + 6);	
						in_temp0 =  _mm_loadu_ps(newIn + ((x) + (y+1-kern_cent_Y+2)*new_size_X));
						mult0 = _mm_mul_ps(kern6,in_temp0);
						out_temp = _mm_add_ps(out_temp, mult0);


						kern7 =  *(newKern  + 7);	
						in_temp1 =  _mm_loadu_ps(newIn + ((x+1) + (y+1-kern_cent_Y+2)*new_size_X));
						mult1 = _mm_mul_ps(kern7,in_temp1);
						out_temp = _mm_add_ps(out_temp, mult1);

						kern8 =  *(newKern + 8);
						in_temp2 =  _mm_loadu_ps(newIn + ((x+2) + (y+1-kern_cent_Y+2)*new_size_X));
						mult2 = _mm_mul_ps(kern8,in_temp2);
						out_temp = _mm_add_ps(out_temp, mult2);

						_mm_storeu_ps(out +(x)+(y)*data_size_X, out_temp);
		}
	}




	///REMAINDER CONVOLUTION LOOP////
	float temp;
	#pragma omp parallel for private(y,x,i,j,temp) num_threads(8)
	for(y = 0; y < data_size_Y; y++){ // the y coordinate of theoutput location we're focusing on
		for(x = data_size_X/4*4; x < data_size_X; x++){ // the x coordinate of the output location we're focusing on
			temp = out[x+y*data_size_X];
			//for(j = -kern_cent_Y; j <= kern_cent_Y; j++){ // kernel unflipped y coordinate
			//	for(i = -kern_cent_X; i <= kern_cent_X; i++){ // kernel unflipped x coordinate
						temp += newKern[(kern_cent_X+i)+(kern_cent_Y+j)*KERNX][0] * newIn[(x+i+1) + (y+j+1)*new_size_X];
			//	}
			//}
			out[x+y*data_size_X] = temp;
		}
	}

	return 1;

}	
